/**
 * Temporal Difference Learning Demo for Game 2048
 * Based on the supplied 2048_sample.cpp; compile with C++11 or later.
 * https://github.com/moporgic/TDL2048-Demo
 *
 * Computer Games and Intelligence (CGI) Lab, NCTU, Taiwan
 * http://www.aigames.nctu.edu.tw
 *
 * References:
 * [1] Szubert, Marcin, and Wojciech Jaśkowski. "Temporal difference learning of n-tuple networks for the game 2048."
 * Computational Intelligence and Games (CIG), 2014 IEEE Conference on. IEEE, 2014.
 * [2] Wu, I-Chen, et al. "Multi-stage temporal difference learning for 2048."
 * Technologies and Applications of Artificial Intelligence. Springer International Publishing, 2014. 366-378.
 * [3] Oka, Kazuto, and Kiminori Matsuzaki. "Systematic selection of n-tuple networks for 2048."
 * International Conference on Computers and Games. Springer International Publishing, 2016.
 */
#include <iostream>
#include <algorithm>
#include <functional>
#include <iterator>
#include <vector>
#include <array>
#include <limits>
#include <numeric>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <iomanip>

/**
 * output streams
 * For debug output, set debug_enabled = true and bind debug to std::cout.
 */
std::ostream& info = std::cout;
std::ostream& error = std::cerr;
const bool debug_enabled = false;
std::ofstream debug_sink;
std::ostream& debug = debug_sink;

/**
 * 64-bit bitboard implementation for 2048
 *
 * index:
 *  0  1  2  3
 *  4  5  6  7
 *  8  9 10 11
 * 12 13 14 15
 *
 * note that the 64-bit value is little endian
 * therefore a board with raw value 0x4312752186532731ull would be
 * +------------------------+
 * |     2     8   128     4|
 * |     8    32    64   256|
 * |     2     4    32   128|
 * |     4     2     8    16|
 * +------------------------+
 *
 */
class board {
public:
	board(uint64_t raw = 0) : raw(raw) {}
	board(const board& b) = default;
	board& operator =(const board& b) = default;
	operator uint64_t() const { return raw; }

	/**
	 * get a 16-bit row
	 */
	int  fetch(int i) const { return ((raw >> (i << 4)) & 0xffff); }
	/**
	 * set a 16-bit row
	 */
	void place(int i, int r) { raw = (raw & ~(0xffffULL << (i << 4))) | (uint64_t(r & 0xffff) << (i << 4)); }
	/**
	 * get a 4-bit tile
	 */
	int  at(int i) const { return (raw >> (i << 2)) & 0x0f; }
	/**
	 * set a 4-bit tile
	 */
	void set(int i, int t) { raw = (raw & ~(0x0fULL << (i << 2))) | (uint64_t(t & 0x0f) << (i << 2)); }

public:
	bool operator ==(const board& b) const { return raw == b.raw; }
	bool operator < (const board& b) const { return raw <  b.raw; }
	bool operator !=(const board& b) const { return !(*this == b); }
	bool operator > (const board& b) const { return b < *this; }
	bool operator <=(const board& b) const { return !(b < *this); }
	bool operator >=(const board& b) const { return !(*this < b); }

private:
	/**
	 * the lookup table for moving board
	 */
	struct lookup {
		int raw; // base row (16-bit raw)
		int left; // left operation
		int right; // right operation
		int score; // merge reward
		bool overflow;

		void init(int r) {
			raw = r;

			int V[4] = { (r >> 0) & 0x0f, (r >> 4) & 0x0f, (r >> 8) & 0x0f, (r >> 12) & 0x0f };
			int L[4] = { V[0], V[1], V[2], V[3] };
			int R[4] = { V[3], V[2], V[1], V[0] }; // mirrored

			score = mvleft(L);
			overflow = std::any_of(L, L + 4, [](int tile) { return tile > 15; });
			left = ((L[0] << 0) | (L[1] << 4) | (L[2] << 8) | (L[3] << 12));

			score = mvleft(R); std::reverse(R, R + 4);
			right = ((R[0] << 0) | (R[1] << 4) | (R[2] << 8) | (R[3] << 12));
		}

		void move_left(uint64_t& raw, int& sc, int i) const {
			if (overflow) throw std::overflow_error("sample bitboard cannot represent a 65536 tile");
			raw |= uint64_t(left) << (i << 4);
			sc += score;
		}

		void move_right(uint64_t& raw, int& sc, int i) const {
			if (overflow) throw std::overflow_error("sample bitboard cannot represent a 65536 tile");
			raw |= uint64_t(right) << (i << 4);
			sc += score;
		}

		static int mvleft(int row[]) {
			int top = 0;
			int tmp = 0;
			int score = 0;

			for (int i = 0; i < 4; i++) {
				int tile = row[i];
				if (tile == 0) continue;
				row[i] = 0;
				if (tmp != 0) {
					if (tile == tmp) {
						tile = tile + 1;
						row[top++] = tile;
						score += (1 << tile);
						tmp = 0;
					} else {
						row[top++] = tmp;
						tmp = tile;
					}
				} else {
					tmp = tile;
				}
			}
			if (tmp != 0) row[top] = tmp;
			return score;
		}

		lookup() {
			static int row = 0;
			init(row++);
		}

		static const lookup& find(int row) {
			static const lookup cache[65536];
			return cache[row];
		}
	};

public:

	/**
	 * reset to initial state (2 random tile on board)
	 */
	void init() { raw = 0; popup(); popup(); }

	/**
	 * add a new random tile on board, or do nothing if the board is full
	 * 2-tile: 70%
	 * 4-tile: 30%
	 */
	void popup() {
		int space[16], num = 0;
		for (int i = 0; i < 16; i++)
			if (at(i) == 0) {
				space[num++] = i;
			}
		if (num)
			set(space[rand() % num], rand() % 10 < 7 ? 1 : 2);
	}

	/**
	 * apply an action to the board
	 * return the reward gained by the action, or -1 if the action is illegal
	 */
	int move(int opcode) {
		switch (opcode) {
		case 0: return move_up();
		case 1: return move_right();
		case 2: return move_down();
		case 3: return move_left();
		default: return -1;
		}
	}

	int move_left() {
		uint64_t move = 0;
		uint64_t prev = raw;
		int score = 0;
		lookup::find(fetch(0)).move_left(move, score, 0);
		lookup::find(fetch(1)).move_left(move, score, 1);
		lookup::find(fetch(2)).move_left(move, score, 2);
		lookup::find(fetch(3)).move_left(move, score, 3);
		raw = move;
		return (move != prev) ? score : -1;
	}
	int move_right() {
		uint64_t move = 0;
		uint64_t prev = raw;
		int score = 0;
		lookup::find(fetch(0)).move_right(move, score, 0);
		lookup::find(fetch(1)).move_right(move, score, 1);
		lookup::find(fetch(2)).move_right(move, score, 2);
		lookup::find(fetch(3)).move_right(move, score, 3);
		raw = move;
		return (move != prev) ? score : -1;
	}
	int move_up() {
		rotate_right();
		int score = move_right();
		rotate_left();
		return score;
	}
	int move_down() {
		rotate_right();
		int score = move_left();
		rotate_left();
		return score;
	}

	/**
	 * swap row and column
	 * +------------------------+       +------------------------+
	 * |     2     8   128     4|       |     2     8     2     4|
	 * |     8    32    64   256|       |     8    32     4     2|
	 * |     2     4    32   128| ----> |   128    64    32     8|
	 * |     4     2     8    16|       |     4   256   128    16|
	 * +------------------------+       +------------------------+
	 */
	void transpose() {
		raw = (raw & 0xf0f00f0ff0f00f0fULL) | ((raw & 0x0000f0f00000f0f0ULL) << 12) | ((raw & 0x0f0f00000f0f0000ULL) >> 12);
		raw = (raw & 0xff00ff0000ff00ffULL) | ((raw & 0x00000000ff00ff00ULL) << 24) | ((raw & 0x00ff00ff00000000ULL) >> 24);
	}

	/**
	 * horizontal reflection
	 * +------------------------+       +------------------------+
	 * |     2     8   128     4|       |     4   128     8     2|
	 * |     8    32    64   256|       |   256    64    32     8|
	 * |     2     4    32   128| ----> |   128    32     4     2|
	 * |     4     2     8    16|       |    16     8     2     4|
	 * +------------------------+       +------------------------+
	 */
	void mirror() {
		raw = ((raw & 0x000f000f000f000fULL) << 12) | ((raw & 0x00f000f000f000f0ULL) << 4)
		    | ((raw & 0x0f000f000f000f00ULL) >> 4) | ((raw & 0xf000f000f000f000ULL) >> 12);
	}

	/**
	 * vertical reflection
	 * +------------------------+       +------------------------+
	 * |     2     8   128     4|       |     4     2     8    16|
	 * |     8    32    64   256|       |     2     4    32   128|
	 * |     2     4    32   128| ----> |     8    32    64   256|
	 * |     4     2     8    16|       |     2     8   128     4|
	 * +------------------------+       +------------------------+
	 */
	void flip() {
		raw = ((raw & 0x000000000000ffffULL) << 48) | ((raw & 0x00000000ffff0000ULL) << 16)
		    | ((raw & 0x0000ffff00000000ULL) >> 16) | ((raw & 0xffff000000000000ULL) >> 48);
	}

	/**
	 * rotate the board clockwise by given times
	 */
	void rotate(int r = 1) {
		switch (((r % 4) + 4) % 4) {
		default:
		case 0: break;
		case 1: rotate_right(); break;
		case 2: reverse(); break;
		case 3: rotate_left(); break;
		}
	}

	void rotate_right() { transpose(); mirror(); } // clockwise
	void rotate_left() { transpose(); flip(); } // counterclockwise
	void reverse() { mirror(); flip(); }

public:

    friend std::ostream& operator <<(std::ostream& out, const board& b) {
		char buff[32];
		out << "+------------------------+" << std::endl;
		for (int i = 0; i < 16; i += 4) {
			snprintf(buff, sizeof(buff), "|%6u%6u%6u%6u|",
				(1 << b.at(i + 0)) & -2u, // use -2u (0xff...fe) to remove the unnecessary 1 for (1 << 0)
				(1 << b.at(i + 1)) & -2u,
				(1 << b.at(i + 2)) & -2u,
				(1 << b.at(i + 3)) & -2u);
			out << buff << std::endl;
		}
		out << "+------------------------+" << std::endl;
		return out;
	}

private:
	uint64_t raw;
};

/**
 * feature and weight table for temporal difference learning
 */
class feature {
public:
	feature(size_t len) : length(len), weight(alloc(len)) {}
	feature(feature&& f) : length(f.length), weight(f.weight) { f.weight = nullptr; }
	feature(const feature& f) = delete;
	feature& operator =(const feature& f) = delete;
	virtual ~feature() { delete[] weight; }

	float& operator[] (size_t i) { return weight[i]; }
	float operator[] (size_t i) const { return weight[i]; }
	size_t size() const { return length; }

public: // should be implemented

	/**
	 * estimate the value of a given board
	 */
	virtual float estimate(const board& b) const = 0;
	/**
	 * update the value of a given board, and return its updated value
	 */
	virtual float update(const board& b, float u) = 0;
	/**
	 * get the name of this feature
	 */
	virtual std::string name() const = 0;

public:

	/**
	 * dump the detail of weight table of a given board
	 */
	virtual void dump(const board& b, std::ostream& out = info) const {
		out << b << "estimate = " << estimate(b) << std::endl;
	}

	friend std::ostream& operator <<(std::ostream& out, const feature& w) {
		std::string name = w.name();
		int len = name.length();
		out.write(reinterpret_cast<char*>(&len), sizeof(int));
		out.write(name.c_str(), len);
		float* weight = w.weight;
		size_t size = w.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size_t));
		out.write(reinterpret_cast<char*>(weight), sizeof(float) * size);
		return out;
	}

	friend std::istream& operator >>(std::istream& in, feature& w) {
		std::string name;
		int len = 0;
		in.read(reinterpret_cast<char*>(&len), sizeof(int));
		if (!in || len < 0 || len > 1024) throw std::runtime_error("invalid feature header");
		name.resize(len);
		in.read(&name[0], len);
		if (name != w.name()) {
			error << "unexpected feature: " << name << " (" << w.name() << " is expected)" << std::endl;
			std::exit(1);
		}
		float* weight = w.weight;
		size_t size = 0;
		in.read(reinterpret_cast<char*>(&size), sizeof(size_t));
		if (size != w.size()) {
			error << "unexpected feature size " << size << "for " << w.name();
			error << " (" << w.size() << " is expected)" << std::endl;
			std::exit(1);
		}
		in.read(reinterpret_cast<char*>(weight), sizeof(float) * size);
		if (!in) {
			error << "unexpected end of binary" << std::endl;
			std::exit(1);
		}
		return in;
	}

protected:
	static float* alloc(size_t num) {
		static size_t total = 0;
		static size_t limit = (1 << 30) / sizeof(float); // 1G memory
		try {
			total += num;
			if (total > limit) throw std::bad_alloc();
			return new float[num]();
		} catch (std::bad_alloc&) {
			error << "memory limit exceeded" << std::endl;
			std::exit(-1);
		}
		return nullptr;
	}
	size_t length;
	float* weight;
};

/**
 * the pattern feature
 * including isomorphic (rotate/mirror)
 *
 * index:
 *  0  1  2  3
 *  4  5  6  7
 *  8  9 10 11
 * 12 13 14 15
 *
 * usage:
 *  pattern({ 0, 1, 2, 3 })
 *  pattern({ 0, 1, 2, 3, 4, 5 })
 */
class pattern : public feature {
public:
	pattern(const std::vector<int>& p, int iso = 8) : feature(1 << (p.size() * 4)), iso_last(iso) {
		set_isomorphic(iso);
		if (p.empty()) {
			error << "no pattern defined" << std::endl;
			std::exit(1);
		}

		/**
		 * isomorphic patterns can be calculated by board
		 *
		 * take pattern { 0, 1, 2, 3 } as an example
		 * apply the pattern to the original board (left), we will get 0x1372
		 * if we apply the pattern to the clockwise rotated board (right), we will get 0x2131,
		 * which is the same as applying pattern { 12, 8, 4, 0 } to the original board
		 * { 0, 1, 2, 3 } and { 12, 8, 4, 0 } are isomorphic patterns
		 * +------------------------+       +------------------------+
		 * |     2     8   128     4|       |     4     2     8     2|
		 * |     8    32    64   256|       |     2     4    32     8|
		 * |     2     4    32   128| ----> |     8    32    64   128|
		 * |     4     2     8    16|       |    16   128   256     4|
		 * +------------------------+       +------------------------+
		 *
		 * therefore if we make a board whose value is 0xfedcba9876543210ull (the same as index)
		 * we would be able to use the above method to calculate its 8 isomorphisms
		 */
		for (int i = 0; i < 8; i++) {
			board idx = 0xfedcba9876543210ull;
			if (i >= 4) idx.mirror();
			idx.rotate(i);
			for (int t : p) {
				isomorphic[i].push_back(idx.at(t));
			}
		}
	}
	pattern(const pattern& p) = delete;
	virtual ~pattern() {}
	pattern& operator =(const pattern& p) = delete;

public:

	/**
	 * estimate the value of a given board
	 */
	virtual float estimate(const board& b) const {
		float value = 0;
		for (int i = 0; i < iso_last; ++i)
			value += operator[](indexof(isomorphic[i], b));
		return value;
	}

	/**
	 * update the value of a given board, and return its updated value
	 */
	virtual float update(const board& b, float u) {
		// learning::update already divides by the number of base patterns.
		// Each symmetry is an active feature, including repeated LUT indices.
		const float delta = u / iso_last;
		for (int i = 0; i < iso_last; ++i)
			operator[](indexof(isomorphic[i], b)) += delta;
		return estimate(b);
	}

	/**
	 * get the name of this feature
	 */
	virtual std::string name() const {
		return std::to_string(isomorphic[0].size()) + "-tuple pattern " + nameof(isomorphic[0]);
	}

public:

	/*
	 * set the isomorphic level of this pattern
	 * 1: no isomorphic
	 * 4: enable rotation
	 * 8: enable rotation and reflection
	 */
	void set_isomorphic(int i = 8) {
		if (i != 1 && i != 4 && i != 8) throw std::invalid_argument("isomorphic must be 1, 4 or 8");
		iso_last = i;
	}

	/**
	 * display the weight information of a given board
	 */
	void dump(const board& b, std::ostream& out = info) const {
		for (int i = 0; i < iso_last; i++) {
			out << "#" << i << ":" << nameof(isomorphic[i]) << "(";
			size_t index = indexof(isomorphic[i], b);
			for (size_t j = 0; j < isomorphic[i].size(); j++) {
				out << std::hex << ((index >> (4 * j)) & 0x0f);
			}
			out << std::dec << ") = " << operator[](index) << std::endl;
		}
	}

protected:

	size_t indexof(const std::vector<int>& patt, const board& b) const {
		size_t index = 0;
		for (size_t i = 0; i < patt.size(); ++i)
			index |= size_t(b.at(patt[i])) << (4 * i);
		return index;
	}

	std::string nameof(const std::vector<int>& patt) const {
		std::stringstream ss;
		ss << std::hex;
		std::copy(patt.cbegin(), patt.cend(), std::ostream_iterator<int>(ss, ""));
		return ss.str();
	}

	std::array<std::vector<int>, 8> isomorphic;
	int iso_last;
};

/**
 * before state and after state wrapper
 */
class state {
public:
	state(int opcode = -1)
		: opcode(opcode), score(-1), esti(-std::numeric_limits<float>::max()) {}
	state(const board& b, int opcode = -1)
		: opcode(opcode), score(-1), esti(-std::numeric_limits<float>::max()) { assign(b); }
	state(const state& st) = default;
	state& operator =(const state& st) = default;

public:
	board after_state() const { return after; }
	board before_state() const { return before; }
	float value() const { return esti; }
	int reward() const { return score; }
	int action() const { return opcode; }

	void set_before_state(const board& b) { before = b; }
	void set_after_state(const board& b) { after = b; }
	void set_value(float v) { esti = v; }
	void set_reward(int r) { score = r; }
	void set_action(int a) { opcode = a; }

public:
	bool operator ==(const state& s) const {
		return (opcode == s.opcode) && (before == s.before) && (after == s.after) && (esti == s.esti) && (score == s.score);
	}
	bool operator < (const state& s) const {
		if (before != s.before) throw std::invalid_argument("state::operator<");
		return esti < s.esti;
	}
	bool operator !=(const state& s) const { return !(*this == s); }
	bool operator > (const state& s) const { return s < *this; }
	bool operator <=(const state& s) const { return !(s < *this); }
	bool operator >=(const state& s) const { return !(*this < s); }

public:

	/**
	 * assign a state (before state), then apply the action (defined in opcode)
	 * return true if the action is valid for the given state
	 */
	bool assign(const board& b) {
		if (debug_enabled) debug << "assign " << name() << std::endl << b;
		after = before = b;
		score = after.move(opcode);
		esti = score;
		return score != -1;
	}

	/**
	 * call this function after initialization (assign, set_value, etc)
	 *
	 * the state is invalid if
	 *  estimated value becomes to NaN (wrong learning rate?)
	 *  invalid action (cause after == before or score == -1)
	 */
	bool is_valid() const {
		if (std::isnan(esti)) {
			error << "numeric exception" << std::endl;
			std::exit(1);
		}
		return after != before && opcode != -1 && score != -1;
	}

	const char* name() const {
		static const char* opname[4] = { "up", "right", "down", "left" };
		return (opcode >= 0 && opcode < 4) ? opname[opcode] : "none";
	}

    friend std::ostream& operator <<(std::ostream& out, const state& st) {
		out << "moving " << st.name() << ", reward = " << st.score;
		if (st.is_valid()) {
			out << ", value = " << st.esti << std::endl << st.after;
		} else {
			out << " (invalid)" << std::endl;
		}
		return out;
	}
private:
	board before;
	board after;
	int opcode;
	int score;
	float esti;
};

class learning {
public:
	learning() {}
	~learning() { for (feature* feat : feats) delete feat; }
	learning(const learning&) = delete;
	learning& operator=(const learning&) = delete;

	/**
	 * add a feature into tuple networks
	 *
	 * note that feats is std::vector<feature*>,
	 * therefore you need to keep all the instances somewhere
	 */
	void add_feature(feature* feat) {
		feats.push_back(feat);

		info << feat->name() << ", size = " << feat->size();
		size_t usage = feat->size() * sizeof(float);
		if (usage >= (1 << 30)) {
			info << " (" << (usage >> 30) << "GB)";
		} else if (usage >= (1 << 20)) {
			info << " (" << (usage >> 20) << "MB)";
		} else if (usage >= (1 << 10)) {
			info << " (" << (usage >> 10) << "KB)";
		}
		info << std::endl;
	}

	/**
	 * accumulate the total value of given state
	 */
	float estimate(const board& b) const {
		if (debug_enabled) debug << "estimate " << std::endl << b;
		float value = 0;
		for (feature* feat : feats) {
			value += feat->estimate(b);
		}
		return value;
	}

	/**
	 * update the value of given state and return its new value
	 */
	float update(const board& b, float u) const {
		if (debug_enabled) debug << "update " << " (" << u << ")" << std::endl << b;
		if (feats.empty()) throw std::logic_error("no features configured");
		float u_split = u / feats.size();
		float value = 0;
		for (feature* feat : feats) {
			value += feat->update(b, u_split);
		}
		return value;
	}

	/**
	 * select a best move of a before state b
	 *
	 * return should be a state whose
	 *  before_state() is b
	 *  after_state() is b's best successor (after state)
	 *  action() is the best action
	 *  reward() is the reward of performing action()
	 *  value() is the action score: immediate reward plus continuation value
	 *
	 * you may simply return state() if no valid move
	 */
	static bool terminal(const board& b) {
		for (int a = 0; a < 4; ++a) {
			board next = b;
			if (next.move(a) != -1) return false;
		}
		return true;
	}

	// Expected V(s'') over every empty cell and both popup tile values.
	float expected_next_value(const board& after) const {
		int empty = 0;
		float value = 0;
		for (int i = 0; i < 16; ++i) {
			if (after.at(i) != 0) continue;
			++empty;
			for (int tile = 1; tile <= 2; ++tile) {
				board next = after;
				next.set(i, tile);
				// With another empty cell, at least one slide is legal.
				bool has_empty = false;
				for (int j = 0; j < 16; ++j) has_empty |= next.at(j) == 0;
				if (has_empty || !terminal(next))
					value += (tile == 1 ? 0.7f : 0.3f) * estimate(next);
			}
		}
		if (!empty) throw std::logic_error("legal afterstate has no empty cells");
		return value / empty;
	}

	state select_best_move(const board& b) const {
		state best;
		best.set_before_state(b);
		best.set_after_state(b);
		for (int a = 0; a < 4; ++a) {
			state candidate(a);
			if (!candidate.assign(b)) continue;
			candidate.set_value(candidate.reward() + expected_next_value(candidate.after_state()));
			if (!best.is_valid() || candidate.value() > best.value()) best = candidate;
		}
		return best;
	}

	/**
	 * update the tuple network by an episode
	 *
	 * path is the sequence of states in this episode,
	 * the last entry in path (path.back()) is the final state
	 *
	 * for example, a 2048 games consists of
	 *  (initial) s0 --(a0,r0)--> s0' --(popup)--> s1 --(a1,r1)--> s1' --(popup)--> s2 (terminal)
	 *  where sx is before state, sx' is after state
	 *
	 * its path would be
	 *  { (s0,s0',a0,r0), (s1,s1',a1,r1), (s2,s2,x,-1) }
	 *  where (x,x,x,x) means (before state, after state, action, reward)
	 */
	void update_episode(std::vector<state>& path, float alpha = 0.1) const {
		if (path.empty()) return;
		// The final record is the terminal before-state sentinel, not a move.
		if (path.back().is_valid()) throw std::logic_error("episode lacks terminal record");
		for (size_t i = path.size() - 1; i > 0; --i) {
			const state& current = path[i - 1];
			const state& next = path[i];
			const board trained = current.before_state();
			const float target = current.reward() + (next.is_valid() ? estimate(next.before_state()) : 0.0f);
			const float error = target - estimate(trained);
			update(trained, alpha * error);
		}
	}

	/**
	 * update the statistic, and display the status once in 1000 episodes by default
	 *
	 * the format would be
	 * 1000   mean = 273901  max = 382324
	 *        512     100%   (0.3%)
	 *        1024    99.7%  (0.2%)
	 *        2048    99.5%  (1.1%)
	 *        4096    98.4%  (4.7%)
	 *        8192    93.7%  (22.4%)
	 *        16384   71.3%  (71.3%)
	 *
	 * where (let unit = 1000)
	 *  '1000': current iteration (games trained)
	 *  'mean = 273901': the average score of last 1000 games is 273901
	 *  'max = 382324': the maximum score of last 1000 games is 382324
	 *  '93.7%': 93.7% (937 games) reached 8192-tiles in last 1000 games (a.k.a. win rate of 8192-tile)
	 *  '22.4%': 22.4% (224 games) terminated with 8192-tiles (the largest) in last 1000 games
	 */
	void make_statistic(size_t n, const board& b, int score, int unit = 1000) {
		scores.push_back(score);
		maxtile.push_back(0);
		for (int i = 0; i < 16; i++) {
			maxtile.back() = std::max(maxtile.back(), b.at(i));
		}

		if (n % unit == 0) { // show the training process
			if (scores.size() != size_t(unit) || maxtile.size() != size_t(unit)) {
				error << "wrong statistic size for show statistics" << std::endl;
				std::exit(2);
			}
			int sum = std::accumulate(scores.begin(), scores.end(), 0);
			int max = *std::max_element(scores.begin(), scores.end());
			int stat[16] = { 0 };
			for (int i = 0; i < 16; i++) {
				stat[i] = std::count(maxtile.begin(), maxtile.end(), i);
			}
			float mean = float(sum) / unit;
			float coef = 100.0 / unit;
			info << n;
			info << "\t" "mean = " << mean;
			info << "\t" "max = " << max;
			info << std::endl;
			for (int t = 1, c = 0; c < unit; c += stat[t++]) {
				if (stat[t] == 0) continue;
				int accu = std::accumulate(stat + t, stat + 16, 0);
				info << "\t" << ((1 << t) & -2u) << "\t" << (accu * coef) << "%";
				info << "\t(" << (stat[t] * coef) << "%)" << std::endl;
			}
			scores.clear();
			maxtile.clear();
		}
	}

	/**
	 * display the weight information of a given board
	 */
	void dump(const board& b, std::ostream& out = info) const {
		out << b << "estimate = " << estimate(b) << std::endl;
		for (feature* feat : feats) {
			out << feat->name() << std::endl;
			feat->dump(b, out);
		}
	}

	/**
	 * load the weight table from binary file
	 * you need to define all the features (add_feature(...)) before call this function
	 */
	void load(const std::string& path) {
		std::ifstream in(path.c_str(), std::ios::binary);
		if (!in) throw std::runtime_error("cannot open model: " + path);
		std::string tag;
		std::getline(in, tag);
		if (tag != "RL2048-V1-td_state") throw std::runtime_error("model format or TD variant mismatch");
		size_t count = 0;
		in.read(reinterpret_cast<char*>(&count), sizeof(count));
		if (!in || count != feats.size()) throw std::runtime_error("model feature count mismatch");
		for (feature* feat : feats) in >> *feat;
		if (!in || in.peek() != std::char_traits<char>::eof())
			throw std::runtime_error("invalid model payload");
		info << "loaded = " << path << std::endl;
	}

	void save(const std::string& path) {
		std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
		if (!out) throw std::runtime_error("cannot write model: " + path);
		out << "RL2048-V1-td_state\n";
		size_t count = feats.size();
		out.write(reinterpret_cast<const char*>(&count), sizeof(count));
		for (feature* feat : feats) out << *feat;
		out.close();
		if (!out) throw std::runtime_error("failed writing model: " + path);
		info << "saved = " << path << std::endl;
	}

private:
	std::vector<feature*> feats;
	std::vector<int> scores;
	std::vector<int> maxtile;
};


struct options {
	size_t episodes = 100000, stats = 1000, checkpoint = 0;
	float alpha = 0.1f;
	unsigned seed = unsigned(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	bool eval = false, episodes_set = false;
	std::string load, save, log;
};

static size_t positive_integer(const std::string& text) {
	if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
		throw std::invalid_argument("expected positive integer: " + text);
	unsigned long long n = std::stoull(text);
	if (!n || n > std::numeric_limits<size_t>::max()) throw std::invalid_argument("integer out of range");
	return size_t(n);
}

int main(int argc, const char* argv[]) {
	try {
		options opt;
		for (int i = 1; i < argc; ++i) {
			std::string arg = argv[i];
			if (arg == "--help") {
				info << "td_state: sample-based TD(0) 2048 trainer\n"
				     << "--episodes N  --alpha X  --seed N  --stats N\n"
				     << "--load FILE  --save FILE  --log CSV  --checkpoint-every N\n"
				     << "--eval (requires --load; default 1000 games, no learning)\n";
				return 0;
			}
			if (arg == "--eval") { opt.eval = true; continue; }
			if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
			std::string value = argv[++i];
			if (arg == "--episodes") { opt.episodes = positive_integer(value); opt.episodes_set = true; }
			else if (arg == "--stats") opt.stats = positive_integer(value);
			else if (arg == "--checkpoint-every") opt.checkpoint = positive_integer(value);
			else if (arg == "--alpha") {
				size_t used = 0;
				opt.alpha = std::stof(value, &used);
				if (used != value.size() || !std::isfinite(opt.alpha) || opt.alpha <= 0 || opt.alpha > 1)
					throw std::invalid_argument("alpha must be in (0, 1]");
			} else if (arg == "--seed") {
				if (value == "0") opt.seed = 0;
				else {
					size_t n = positive_integer(value);
					if (n > std::numeric_limits<unsigned>::max()) throw std::invalid_argument("seed out of range");
					opt.seed = unsigned(n);
				}
			} else if (arg == "--load") opt.load = value;
			else if (arg == "--save") opt.save = value;
			else if (arg == "--log") opt.log = value;
			else throw std::invalid_argument("unknown option: " + arg);
		}
		if (opt.eval && opt.load.empty()) throw std::invalid_argument("--eval requires --load");
		if (opt.eval && !opt.episodes_set) opt.episodes = 1000;
		if (opt.eval && (!opt.save.empty() || opt.checkpoint)) throw std::invalid_argument("evaluation does not save weights");
		if (!opt.eval && opt.save.empty()) opt.save = "td_state.bin";
		if (!opt.log.empty() && (opt.log == opt.load || opt.log == opt.save))
			throw std::invalid_argument("log must differ from model paths");
		std::ofstream csv;
		if (!opt.log.empty()) {
			csv.open(opt.log.c_str());
			if (!csv) throw std::runtime_error("cannot open log: " + opt.log);
			csv << "episode,score,max_tile,moves,elapsed_seconds\n";
		}
		learning tdl;
		tdl.add_feature(new pattern({0, 1, 2, 3, 4, 5}));
		tdl.add_feature(new pattern({4, 5, 6, 7, 8, 9}));
		tdl.add_feature(new pattern({0, 1, 2, 4, 5, 6}));
		tdl.add_feature(new pattern({4, 5, 6, 8, 9, 10}));
		if (!opt.load.empty()) tdl.load(opt.load);
		std::srand(opt.seed);
		info << "variant = td_state\nmode = " << (opt.eval ? "eval" : "train")
		     << "\nalpha = " << opt.alpha << "\nepisodes = " << opt.episodes << "\nseed = " << opt.seed << std::endl;
		std::vector<state> path;
		path.reserve(20000);
		const auto start = std::chrono::steady_clock::now();
		long long total_score = 0, window_score = 0;
		size_t total_wins = 0, window_wins = 0, window_count = 0;
		int max_score = 0;
		for (size_t n = 1; n <= opt.episodes; ++n) {
			board b;
			b.init();
			int score = 0, moves = 0;
			path.clear();
			while (true) {
				state best = tdl.select_best_move(b);
				if (!opt.eval) path.push_back(best);
				if (!best.is_valid()) break;
				score += best.reward();
				++moves;
				b = best.after_state();
				b.popup();
			}
			if (!opt.eval) tdl.update_episode(path, opt.alpha);
			int tile = 0;
			for (int i = 0; i < 16; ++i) tile = std::max(tile, b.at(i));
			bool win = tile >= 11;
			total_score += score; window_score += score;
			total_wins += win; window_wins += win; ++window_count;
			max_score = std::max(max_score, score);
			double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
			if (csv.is_open()) {
				csv << n << ',' << score << ',' << (1 << tile) << ',' << moves << ',' << seconds << '\n';
				if (!csv) throw std::runtime_error("failed writing training log");
			}
			if (n % opt.stats == 0 || n == opt.episodes) {
				info << n << "	mean = " << double(window_score) / window_count
				     << "	win2048 = " << 100.0 * window_wins / window_count
				     << "%	elapsed = " << seconds << "s" << std::endl;
				window_count = window_wins = 0; window_score = 0;
				if (csv.is_open()) csv.flush();
			}
			if (!opt.eval && opt.checkpoint && n % opt.checkpoint == 0 && n != opt.episodes)
				tdl.save(opt.save);
		}
		if (!opt.eval) tdl.save(opt.save);
		info << "summary: games = " << opt.episodes << " mean = " << double(total_score) / opt.episodes
		     << " max = " << max_score << " win2048 = " << 100.0 * total_wins / opt.episodes << "%" << std::endl;
		return 0;
	} catch (const std::exception& e) {
		error << "error: " << e.what() << std::endl;
		return 1;
	}
}
