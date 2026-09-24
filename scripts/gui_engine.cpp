// Read-only model inference adapter. The submitted training files stay unchanged.
#define main training_main
#ifdef GUI_TD_STATE
#include "../td_state.cpp"
#else
#include "../td_afterstate.cpp"
#endif
#undef main

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("expected model path");
        // Keep the stdout protocol clean; sample status messages go to stderr.
        std::streambuf* protocol = std::cout.rdbuf(std::cerr.rdbuf());
        learning model;
        model.add_feature(new pattern({0, 1, 2, 3, 4, 5}));
        model.add_feature(new pattern({4, 5, 6, 7, 8, 9}));
        model.add_feature(new pattern({0, 1, 2, 4, 5, 6}));
        model.add_feature(new pattern({4, 5, 6, 8, 9, 10}));
        model.load(argv[1]);
        std::cout.rdbuf(protocol);
        board b;
        int score = 0, moves = 0, last_action = -1, last_reward = 0;
        std::srand(2026);
        b.init();
        auto output = [&]() {
            state best = model.select_best_move(b);
            std::cout << "{\"score\":" << score << ",\"moves\":" << moves
                      << ",\"terminal\":" << (best.is_valid() ? "false" : "true")
                      << ",\"best_action\":" << best.action()
                      << ",\"last_action\":" << last_action << ",\"reward\":" << last_reward
                      << ",\"board\":[";
            for (int i = 0; i < 16; ++i) {
                if (i) std::cout << ',';
                std::cout << ((1 << b.at(i)) & -2u);
            }
            std::cout << "],\"values\":[";
            for (int a = 0; a < 4; ++a) {
                if (a) std::cout << ',';
                state candidate(b, a);
                if (!candidate.is_valid()) { std::cout << "null"; continue; }
#ifdef GUI_TD_STATE
                float value = candidate.reward() + model.expected_next_value(candidate.after_state());
#else
                float value = candidate.reward() + model.estimate(candidate.after_state());
#endif
                std::cout << std::setprecision(9) << value;
            }
            std::cout << "]}" << std::endl;
        };
        output();
        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream command(line);
            std::string op;
            command >> op;
            if (op == "quit") break;
            if (op == "reset") {
                unsigned seed;
                if (!(command >> seed)) throw std::invalid_argument("reset requires a seed");
                std::srand(seed); b.init(); score = moves = last_reward = 0; last_action = -1;
            } else if (op == "ai" || op == "move") {
                int action = -1;
                if (op == "ai") action = model.select_best_move(b).action();
                else if (!(command >> action) || action < 0 || action > 3)
                    throw std::invalid_argument("move requires action 0..3");
                const int reward = b.move(action);
                last_reward = reward;
                last_action = action;
                if (reward >= 0) { score += reward; ++moves; b.popup(); }
            } else if (op != "state") throw std::invalid_argument("unknown command");
            output();
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
