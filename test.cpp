#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#define ep emplace_back
#define all(x) begin(x), end(x)

struct Bandit {
	std::vector<double> mu; // 真实均值 μ_i
	double best_mu;

	explicit Bandit(std::vector<double> means)
	: mu(std::move(means)), best_mu(*std::max_element(all(mu))) {  }
	virtual ~Bandit() = default;

	int K() const { return static_cast<int>(mu.size()); }
	virtual double step(int id, std::mt19937_64 &rng) = 0;
};
struct Bernoulli_Bandit : Bandit {
	std::vector<std::bernoulli_distribution> arms;
	explicit Bernoulli_Bandit(std::vector<double> means) : Bandit(std::move(means)) {
		for(double m : mu) arms.ep(m);
	}
	double step(int id, std::mt19937_64 &rng) override { return arms[id](rng) ? 1.0 : 0.0; }
}; // 伯努利分布环境
struct Gaussian_Bandit : Bandit {
	double sigma;
	Gaussian_Bandit(std::vector<double> means, double sigma = 1.0) : Bandit(std::move(means)), sigma(sigma) {  }
	double step(int id, std::mt19937_64 &rng) override {
		std::normal_distribution<double> noise(0.0, sigma);
		return mu[id] + noise(rng);
	}
}; // 正态分布环境

class Solver {
protected:
	std::vector<uint64_t> N_; // 每个臂被选了多少次
	std::vector<double> S_; // 每个臂的累计贡献

	double mean(int i) const {
		static const double oo = std::numeric_limits<double>::infinity();
		return N_[i] == 0 ? +oo : S_[i] / static_cast<double>(N_[i]); // 三个算法中没被选过的都会优先选一次，所以直接设为 +oo
	}
public:
	explicit Solver(int K) : N_(K, 0), S_(K, 0.0) {  }
	virtual ~Solver() = default;
	
	virtual int select(uint64_t t, std::mt19937_64 &rng) = 0;
	void update(int id, double r) { ++N_[id], S_[id] += r; }
	std::vector<uint64_t> counts() const { return N_; }

	void set(const std::vector<uint64_t> &N, const std::vector<double> &S) { N_ = N, S_ = S; }
	int random_select_max(const std::vector<double> &score, std::mt19937_64 &rng) const {
		double Max = *std::max_element(all(score));
		std::vector<int> vec;
		for(int i = 0; i < static_cast<int>(N_.size()); ++i)
			if(score[i] == Max) vec.ep(i);
		std::uniform_int_distribution<int> rnd(0, static_cast<int>(vec.size()) - 1);
		return vec[rnd(rng)];
	}
	std::vector<double> estimate() const {
		std::vector<double> e(N_.size());
		for(int i = 0; i < static_cast<int>(N_.size()); ++i) e[i] = mean(i);
		return e;
	}
};

class Greedy : public Solver {
public:
	using Solver::Solver;
	int select(uint64_t t, std::mt19937_64 &rng) override {
		return random_select_max(estimate(), rng);
	}
};

class Epsilon_Greedy : public Solver {
private:
	double eps_;
	int K() const { return static_cast<int>(N_.size()); }
public:
	Epsilon_Greedy(int K, double eps) : Solver(K), eps_(eps) {  }
	int select(uint64_t t, std::mt19937_64 &rng) override {
		static const double oo = std::numeric_limits<double>::infinity();
		auto score = estimate();
		double Max = *std::max_element(all(score));
		std::uniform_real_distribution<double> u(0.0, 1.0);
		if(Max == +oo || u(rng) >= eps_) return random_select_max(score, rng);
		return std::uniform_int_distribution<int>(0, K() - 1)(rng);
	}
};

class UCB : public Solver {
private:
	double c_;
public:
	UCB(int K, double c) : Solver(K), c_(c) {  }
	int select(uint64_t t, std::mt19937_64 &rng) override {
		const double oo = std::numeric_limits<double>::infinity();
		auto u = estimate();
		for(int i = 0; i < static_cast<int>(N_.size()); ++i) {
			if(u[i] == +oo) continue;
			u[i] += c_ * std::sqrt(std::log(static_cast<double>(t)) / static_cast<double>(N_[i]));
		}
		return random_select_max(u, rng);
	}
};

struct Episode {
	std::vector<double> regret;
	std::vector<uint64_t> counts;
};

Episode run(Bandit &env, Solver &sol, int T, std::mt19937_64 &rng) {
	Episode cur;
	cur.regret.resize(T);
	double tot = 0.0;
	for(int t = 1; t <= T; ++t) {
		int id = sol.select(t, rng);
		double r = env.step(id, rng);
		sol.update(id, r);
		tot += env.best_mu - env.mu[id]; // 累加 Delta_i
		cur.regret[t - 1] = tot;
	}
	cur.counts = sol.counts();
	return cur;
}

struct Experiment {
	std::vector<double> mean_regret;
	std::vector<double> mean_counts;
};

using Solver_Factory = std::function<std::unique_ptr<Solver>(int)>;
using Env_Factory = std::function<std::unique_ptr<Bandit>(const std::vector<double>&)>;

Experiment experiment(const std::vector<double> &means,
					  const Solver_Factory& sol_factory,
					  int T, int runs,
					  std::uint64_t seed = 114514,
					  const Env_Factory &env_factory = [](const std::vector<double> &m) -> std::unique_ptr<Bandit> {
						return std::make_unique<Bernoulli_Bandit>(m);
					  }) {
	const int K = static_cast<int>(means.size());
	Experiment cur;
	cur.mean_regret.assign(T, 0.0);
	cur.mean_counts.assign(K, 0.0);
	for(int k = 0; k < runs; ++k) {
		std::mt19937_64 rng(seed + k);
		auto env = env_factory(means);
		auto sol = sol_factory(env -> K());
		const Episode now = run(*env, *sol, T, rng);
		for(int t = 0; t < T; ++t)
			cur.mean_regret[t] += now.regret[t] / static_cast<double>(runs);
		for(int i = 0; i < K; ++i)
			cur.mean_counts[i] += now.counts[i] / static_cast<double>(runs);
	}
	return cur;
}

int main() {
	constexpr int runs = 100;
	const std::vector<double> means_bernoulli = { 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8 };
	const std::vector<double> means_gaussian = { 0.705, 1.294, -0.574, 0.398, -1.907, 1.119, -0.724, -1.492, 0.015, -0.309 };
	/*
		正态分布的均值使用
		```
		std::mt19937_64 rng(42);
		std::normal_distribution<double> N01(0.0, 1.0);
		std::vector<double> means(10);
		for (double& m : means) m = N01(rng);
		```
		生成
	*/

	const std::vector<std::pair<std::string, Solver_Factory>> algos = {
		{"greedy", 			[](int K) { return std::make_unique<Greedy>(K); }},
		{"eps-greedy 0.05",   [](int K) { return std::make_unique<Epsilon_Greedy>(K, 0.05); }},
		{"eps-greedy 0.1",    [](int K) { return std::make_unique<Epsilon_Greedy>(K, 0.1); }},
		{"UCB 0.5",         [](int K) { return std::make_unique<UCB>(K, 0.5); }},
		{"UCB sqrt2",       [](int K) { return std::make_unique<UCB>(K, sqrt(2.0)); }},
		{"UCB 2", 		    [](int K) { return std::make_unique<UCB>(K, 2); }},
	};

	auto run_exp = [&](bool type) -> void {
		if(!type) std::cout << "===== Bernoulli Distribution: K = 10, runs = " << runs << " =====\nT\t";
		else std::cout << "===== Gaussian Distribution: K = 10, sigma = 1, runs = " << runs << " =====\nT\t";
		for(auto &[name, func] : algos) std::cout << name << '\t';
		std::cout << '\n';
		const Env_Factory env_f = [type](const std::vector<double> &m) -> std::unique_ptr<Bandit> {
			if(type) return std::make_unique<Gaussian_Bandit>(m, 1.0);
			return std::make_unique<Bernoulli_Bandit>(m);
		};
		for(int T : {1000, 5000, 20000}) {
			std::cout << T << '\t';
			for(auto &[name, func] : algos) {
				const Experiment exp_ = experiment(type ? means_gaussian : means_bernoulli, func, T, runs, 114514, env_f);
				std::cout << exp_.mean_regret.back() << '\t';
			}
			std::cout << '\n';
		}
	};
	run_exp(0); // 伯努利分布环境
	run_exp(1); // 正态分布环境

	// 验证 regret R_T = \sum_i \Delta_i N_i(T)
	std::mt19937_64 rng(67);
	Bernoulli_Bandit env(means_bernoulli);
	UCB sol(env.K(), 0.5);
	const Episode now = run(env, sol, 5000, rng);
	double rhs = 0.0;
	for(int i = 0; i < env.K(); ++i)
		rhs += (env.best_mu - env.mu[i]) * static_cast<double>(now.counts[i]);
	std::cout << "[test] regret: R_T = " << now.regret.back() << ", ";
	std::cout << "sum Delta_i N_i(T) = " << rhs << ", ";
	std::cout << "diff = " << std::abs(now.regret.back() - rhs) << '\n';
}