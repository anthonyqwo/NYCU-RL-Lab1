// Compile twice: default after-state; -DTEST_STATE for state learning.
#define main training_main
#ifdef TEST_STATE
#include "../td_state.cpp"
#else
#include "../td_afterstate.cpp"
#endif
#undef main
#include <map>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static void near(float actual, float expected, const char* message) {
    require(std::fabs(actual - expected) < 0.0005f, message);
}

struct exposed_pattern : pattern {
    using pattern::pattern;
    using pattern::indexof;
};

// Independent exact table: isolates TD bookkeeping from function approximation.
struct exact_feature : feature {
    std::map<uint64_t, float> values;
    exact_feature() : feature(1) {}
    float estimate(const board& b) const override {
        auto it = values.find(uint64_t(b));
        return it == values.end() ? 0.0f : it->second;
    }
    float update(const board& b, float u) override { return values[uint64_t(b)] += u; }
    std::string name() const override { return "exact-test"; }
};

static board blocked_board() {
    // Distinct neighbours, including at wrap-free row/column boundaries.
    board b;
    for (int i = 0; i < 16; ++i) b.set(i, ((i / 4 + i % 4) % 2) + 1);
    return b;
}

int main() {
    try {
        board b;
        b.set(0, 1); b.set(1, 1); b.set(2, 1); b.set(3, 1);
        require(b.move(3) == 8, "merge reward");
        require(b.at(0) == 2 && b.at(1) == 2 && b.at(2) == 0, "single merge per tile");
        board rotated = b;
        rotated.rotate(4);
        require(rotated == b, "full rotation");
        rotated.rotate(); rotated.rotate(); rotated.rotate(); rotated.rotate();
        require(rotated == b, "four rotations");
        require(learning::terminal(blocked_board()), "terminal detection");
        board overflow; overflow.set(0, 15); overflow.set(1, 15);
        bool rejected = false;
        try { overflow.move(3); } catch (const std::overflow_error&) { rejected = true; }
        require(rejected, "bitboard overflow must not corrupt neighbouring tiles");

        exposed_pattern plain({0, 1, 2}, 1);
        board digits; digits.set(0, 1); digits.set(1, 2); digits.set(2, 3);
        require(plain.indexof({0, 1, 2}, digits) == 0x321, "tuple digit ordering");
        near(plain.update(digits, 2.0f), 2.0f, "single feature update");
        near(plain.estimate(digits), 2.0f, "single feature estimate");
        exposed_pattern sym({0, 1}, 8);
        sym.update(digits, 0.8f);
        float symmetric_value = sym.estimate(digits);
        for (int i = 0; i < 8; ++i) {
            board transformed = digits;
            if (i >= 4) transformed.mirror();
            transformed.rotate(i);
            near(sym.estimate(transformed), symmetric_value, "symmetry invariance");
        }
        exposed_pattern duplicate({0}, 8);
        near(duplicate.update(board(), 0.8f), 6.4f, "shared symmetric index multiplicity");
        near(duplicate[0], 0.8f, "shared LUT receives every active feature gradient");

        learning expected;
        exact_feature* table = new exact_feature;
        expected.add_feature(table);
        board after;
        for (int i = 0; i < 16; ++i) after.set(i, 3);
        after.set(0, 0); after.set(15, 0);
        for (int cell : {0, 15}) for (int tile = 1; tile <= 2; ++tile) {
            board next = after; next.set(cell, tile);
            table->values[uint64_t(next)] = float(cell + 10 * tile);
        }
        near(expected.expected_next_value(after), 20.5f, "popup probabilities and uniform cells");
        board almost = blocked_board(); almost.set(0, 0);
        board dead = almost; dead.set(0, 1);
        board alive = almost; alive.set(0, 2);
        table->values[uint64_t(dead)] = 1000;
        table->values[uint64_t(alive)] = 10;
        near(expected.expected_next_value(almost), 3.0f, "terminal popup value is zero");
        state no_move = expected.select_best_move(blocked_board());
        require(!no_move.is_valid(), "no legal action");
        require(no_move.before_state() == blocked_board(), "terminal sentinel retains board");

        // Check the complete action ranking against explicit popup enumeration.
        board decision; decision.set(0, 1); decision.set(1, 1); decision.set(6, 2);
        for (int a = 0; a < 4; ++a) {
            board next = decision;
            if (next.move(a) < 0) continue;
            table->values[uint64_t(next)] = float(10 * a);
            for (int cell = 0; cell < 16; ++cell) if (next.at(cell) == 0) {
                for (int tile = 1; tile <= 2; ++tile) {
                    board popped = next; popped.set(cell, tile);
                    table->values[uint64_t(popped)] = float((uint64_t(popped) % 97) + a);
                }
            }
        }
        float best_score = -1; int best_action = -1;
        for (int a = 0; a < 4; ++a) {
            board next = decision;
            int reward = next.move(a);
            if (reward < 0) continue;
            float continuation = 0;
#ifdef TEST_STATE
            int cells = 0;
            for (int cell = 0; cell < 16; ++cell) if (next.at(cell) == 0) {
                ++cells;
                board two = next, four = next; two.set(cell, 1); four.set(cell, 2);
                continuation += 0.7f * table->estimate(two) + 0.3f * table->estimate(four);
            }
            continuation /= cells;
#else
            continuation = table->estimate(next);
#endif
            if (reward + continuation > best_score) { best_score = reward + continuation; best_action = a; }
        }
        state chosen = expected.select_best_move(decision);
        require(chosen.action() == best_action, "variant-specific action ranking");
        near(chosen.value(), best_score, "variant-specific action score");

        // Nonterminal backup where re-selection must override recorded action/value.
        learning learner;
        exact_feature* lut = new exact_feature;
        learner.add_feature(lut);
        board first; first.set(0, 1); first.set(1, 1);
        state current(first, 3); // reward 4, afterstate has one 4-tile
        board next_board = current.after_state(); next_board.set(5, 1);
        state recorded(next_board, 0);
        recorded.set_value(9999); // stale trajectory value must never be a target
        state greedy_next(next_board, 1);
        lut->values[uint64_t(greedy_next.after_state())] = 50;
        lut->values[uint64_t(next_board)] = 20;
        state terminal;
        terminal.set_before_state(blocked_board()); terminal.set_after_state(blocked_board());
        std::vector<state> path = {current, recorded, terminal};
        learner.update_episode(path, 0.1f);
#ifdef TEST_STATE
        // Reverse update changes next V from 20 to 18 (recorded move reward 0).
        near(lut->estimate(next_board), 18.0f, "terminal state backup");
        near(lut->estimate(first), 2.2f, "state uses sampled next state and current weights");
#else
        near(lut->estimate(recorded.after_state()), 0.0f, "terminal afterstate target");
        near(lut->estimate(current.after_state()), 5.0f, "afterstate reselects greedy action");
#endif
        // Zero initialization must select the highest immediate merge reward.
        learning zero;
        zero.add_feature(new exact_feature);
        board reward_board; reward_board.set(0, 1); reward_board.set(1, 1); reward_board.set(4, 2);
        state selected = zero.select_best_move(reward_board);
        require(selected.is_valid() && selected.reward() == 4, "reward participates in action choice");

        // Round-trip the real tuple serialization with small LUTs.
        learning original, restored;
        original.add_feature(new pattern({0, 1}, 8));
        restored.add_feature(new pattern({0, 1}, 8));
        original.update(digits, 1.25f);
#ifdef TEST_STATE
        const char* model_path = "build/test-state.bin";
#else
        const char* model_path = "build/test-afterstate.bin";
#endif
        original.save(model_path);
        restored.load(model_path);
        near(restored.estimate(digits), original.estimate(digits), "model serialization");
        std::remove(model_path);
        info << "All TD tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        error << "TEST FAILED: " << e.what() << '\n';
        return 1;
    }
}
