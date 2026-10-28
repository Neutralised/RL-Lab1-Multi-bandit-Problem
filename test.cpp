#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

#define ep emplace_back
#define all(x) begin(x), end(x)
#ifdef _WIN32
#define MAKE_DIR(p) _mkdir(p)
#else
#define MAKE_DIR(p) mkdir(p, 0755)
#endif

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
		(void)t; // Greedy 不需要轮次信息
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
		(void)t;
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

class Reducing_Epsilon_Greedy : public Solver {
private:
	double eps_;
	int K() const { return static_cast<int>(N_.size()); }
public:
	Reducing_Epsilon_Greedy(int K, double eps) : Solver(K), eps_(eps) {  }
	int select(uint64_t t, std::mt19937_64 &rng) override {
		static const double oo = std::numeric_limits<double>::infinity();
		auto score = estimate();
		double Max = *std::max_element(all(score));
		std::uniform_real_distribution<double> u(0.0, 1.0);
		// ε_t = ε_0 / sqrt(t)：探索概率随时间衰减
		if(Max == +oo || u(rng) >= eps_ / std::sqrt(static_cast<double>(t))) return random_select_max(score, rng);
		return std::uniform_int_distribution<int>(0, K() - 1)(rng);
	}
};

// ============================ 单条轨迹 ============================
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

// ============================ 多次重复的统计量 ============================
struct Experiment {
	std::vector<double> mean_regret; // 每个时刻的平均累积 regret
	std::vector<double> mean_counts; // 每个臂的平均选择次数
};

struct Experiment_Full {
	std::vector<double> mean_regret; // 每个时刻的平均累积 regret
	std::vector<double> se_regret;   // 每个时刻的标准误（用于画置信带）
	std::vector<double> opt_ratio;   // 每个时刻的"累计选中最优臂的比例"的平均
	std::vector<double> mean_counts; // 每个臂的平均选择次数
};

using Solver_Factory = std::function<std::unique_ptr<Solver>(int)>;
using Env_Factory = std::function<std::unique_ptr<Bandit>(const std::vector<double>&)>;

Experiment_Full experiment_full(const std::vector<double> &means,
								const Solver_Factory &sol_factory,
								int T, int runs,
								std::uint64_t seed,
								const Env_Factory &env_factory) {
	const int K = static_cast<int>(means.size());
	const double best = *std::max_element(all(means));
	std::vector<char> is_opt(K, 0);
	for(int i = 0; i < K; ++i) is_opt[i] = (means[i] == best);

	Experiment_Full cur;
	cur.mean_regret.assign(T, 0.0);
	cur.se_regret.assign(T, 0.0);
	cur.opt_ratio.assign(T, 0.0);
	cur.mean_counts.assign(K, 0.0);
	std::vector<double> sq(T, 0.0); // 平方和

	for(int k = 0; k < runs; ++k) {
		std::mt19937_64 rng(seed + k);
		auto env = env_factory(means);
		auto sol = sol_factory(env -> K());

		std::vector<double> reg(T);
		double tot = 0.0;
		long long opt = 0;
		for(int t = 1; t <= T; ++t) {
			int id = sol -> select(t, rng);
			double r = env -> step(id, rng);
			sol -> update(id, r);
			tot += env -> best_mu - env -> mu[id];
			reg[t - 1] = tot;
			if(is_opt[id]) ++opt;
			cur.opt_ratio[t - 1] += static_cast<double>(opt) / t / runs;
		}
		for(int t = 0; t < T; ++t) {
			cur.mean_regret[t] += reg[t] / runs;
			sq[t] += reg[t] * reg[t] / runs;
		}
		auto cnt = sol -> counts();
		for(int i = 0; i < K; ++i) cur.mean_counts[i] += cnt[i] / static_cast<double>(runs);
	}
	for(int t = 0; t < T; ++t) { // SE = sqrt((E[x^2] - E[x]^2) / runs)
		const double var = sq[t] - cur.mean_regret[t] * cur.mean_regret[t];
		cur.se_regret[t] = var > 0.0 ? std::sqrt(var / runs) : 0.0;
	}
	return cur;
}

// 旧接口：只返回均值（保留下来供对照，内部复用 experiment_full）
Experiment experiment(const std::vector<double> &means,
					  const Solver_Factory& sol_factory,
					  int T, int runs,
					  std::uint64_t seed = 114514,
					  const Env_Factory &env_factory = [](const std::vector<double> &m) -> std::unique_ptr<Bandit> {
						return std::make_unique<Bernoulli_Bandit>(m);
					  }) {
	const Experiment_Full full = experiment_full(means, sol_factory, T, runs, seed, env_factory);
	return Experiment{ full.mean_regret, full.mean_counts };
}

// ============================ CSV 输出 ============================
struct Csv_Writer {
	std::ofstream curves, summary, counts;

	Csv_Writer() {
		MAKE_DIR("results");
		curves.open("results/curves.csv");
		summary.open("results/summary.csv");
		counts.open("results/counts.csv");
		for(std::ofstream *f : { &curves, &summary, &counts }) {
			(*f) << std::fixed << std::setprecision(6);
		}
		curves << "experiment,env,algo,param_name,param_value,T,runs,t,mean_regret,se_regret,opt_ratio\n";
		summary << "experiment,env,algo,param_name,param_value,T,runs,K,mu_star,delta_min,mean_delta,"
				   "final_regret,regret_per_t,opt_ratio_final\n";
		counts << "experiment,env,algo,param_name,param_value,T,arm,mu,count\n";
	}
	bool ok() const { return curves.is_open() && summary.is_open() && counts.is_open(); }
};

// 把一条实验的结果写进三个 CSV：逐步曲线、终值汇总、每臂选择次数
void record(Csv_Writer &csv, const std::string &exp, const std::string &env_name,
			const std::string &algo, const std::string &param_name, double param_value,
			const std::vector<double> &means, int T, int runs,
			const Experiment_Full &e, int max_points = 200) {
	const int K = static_cast<int>(means.size());
	const double best = *std::max_element(all(means));
	double delta_min = std::numeric_limits<double>::infinity();
	double delta_sum = 0.0;

	for(int i = 0; i < K; ++i) {
		const double d = best - means[i];
		if(d > 0.0) {
			delta_min = std::min(delta_min, d);
			delta_sum += d;
		}
	}
	if(!std::isfinite(delta_min)) delta_min = 0.0; // 所有臂并列时

	const int stride = std::max(1, T / max_points);
	for(int t = 1; t <= T; ++t) {
		if(t % stride != 0 && t != T) continue; // 只保留 ~max_points 个采样点，避免 CSV 过大
		csv.curves << exp << ',' << env_name << ',' << algo << ',' << param_name << ',' << param_value << ','
				   << T << ',' << runs << ',' << t << ','
				   << e.mean_regret[t - 1] << ',' << e.se_regret[t - 1] << ',' << e.opt_ratio[t - 1] << '\n';
	}

	csv.summary << exp << ',' << env_name << ',' << algo << ',' << param_name << ',' << param_value << ','
				<< T << ',' << runs << ',' << K << ',' << best << ',' << delta_min << ','
				<< (K > 1 ? delta_sum / (K - 1) : 0.0) << ','
				<< e.mean_regret.back() << ',' << e.mean_regret.back() / T << ','
				<< e.opt_ratio.back() << '\n';

	for(int i = 0; i < K; ++i)
		csv.counts << exp << ',' << env_name << ',' << algo << ',' << param_name << ',' << param_value << ','
				   << T << ',' << i << ',' << means[i] << ',' << e.mean_counts[i] << '\n';
}

// ============================ 实验配置 ============================
main() {
	constexpr int runs = 200;   // 每个配置重复的次数
	constexpr std::uint64_t seed = 114514;

	const std::vector<double> means_bernoulli = { 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8 };
	// 正态分布的真值：由 std::mt19937_64 rng(42) + N(0,1) 抽样得到
	const std::vector<double> means_gaussian = { 0.705, 1.294, -0.574, 0.398, -1.907, 1.119, -0.724, -1.492, 0.015, -0.309 };

	const Env_Factory bern = [](const std::vector<double> &m) -> std::unique_ptr<Bandit> {
		return std::make_unique<Bernoulli_Bandit>(m);
	};
	const Env_Factory gauss = [](const std::vector<double> &m) -> std::unique_ptr<Bandit> {
		return std::make_unique<Gaussian_Bandit>(m, 1.0);
	};

	// 主实验用到的算法集合
	const std::vector<std::pair<std::string, Solver_Factory>> algos = {
		{"greedy",                     [](int K) { return std::make_unique<Greedy>(K); }},
		{"eps-greedy 0.01",            [](int K) { return std::make_unique<Epsilon_Greedy>(K, 0.01); }},
		{"eps-greedy 0.05",            [](int K) { return std::make_unique<Epsilon_Greedy>(K, 0.05); }},
		{"eps-greedy 0.1",             [](int K) { return std::make_unique<Epsilon_Greedy>(K, 0.1); }},
		{"reducing-eps-greedy 0.05",   [](int K) { return std::make_unique<Reducing_Epsilon_Greedy>(K, 0.05); }},
		{"UCB 0.5",                    [](int K) { return std::make_unique<UCB>(K, 0.5); }},
		{"UCB sqrt2",                  [](int K) { return std::make_unique<UCB>(K, std::sqrt(2.0)); }},
		{"UCB 2",                      [](int K) { return std::make_unique<UCB>(K, 2.0); }},
	};
	// T 扫描与 gap 扫描只用三个代表算法，控制运行时间
	const std::vector<std::pair<std::string, Solver_Factory>> algos_few = {
		{"greedy",                     [](int K) { return std::make_unique<Greedy>(K); }},
		{"eps-greedy 0.05",            [](int K) { return std::make_unique<Epsilon_Greedy>(K, 0.05); }},
		{"UCB sqrt2",                  [](int K) { return std::make_unique<UCB>(K, std::sqrt(2.0)); }},
	};

	Csv_Writer csv;
	if(!csv.ok()) {
		std::cerr << "无法写入 results/*.csv，请确认当前目录可写\n";
		return 1;
	}

	// ---------- E1：基准环境下的算法对比（K=10，T=2000） ----------
	for(auto &[name, f] : algos) {
		const auto e = experiment_full(means_bernoulli, f, 2000, runs, seed, bern);
		record(csv, "E1_algorithms", "bernoulli", name, "-", 0, means_bernoulli, 2000, runs, e);
	}
	// ---------- E6：同一组算法在高斯环境（sigma=1）下的对比 ----------
	for(auto &[name, f] : algos) {
		const auto e = experiment_full(means_gaussian, f, 2000, runs, seed, gauss);
		record(csv, "E6_gaussian", "gaussian", name, "-", 0, means_gaussian, 2000, runs, e);
	}

	// ---------- E2：只改 T ----------
	for(int T : {200, 500, 1000, 2000, 5000, 10000}) {
		for(auto &[name, f] : algos_few) {
			const auto e = experiment_full(means_bernoulli, f, T, runs, seed, bern);
			record(csv, "E2_horizon", "bernoulli", name, "T", T, means_bernoulli, T, runs, e);
		}
	}

	// ---------- E3：只改均值差距（K=2，mu=(0.6, 0.6-gap)） ----------
	for(double gap : {0.3, 0.2, 0.1, 0.05, 0.02, 0.01}) {
		const std::vector<double> means = { 0.6, 0.6 - gap };
		for(auto &[name, f] : algos_few) {
			const auto e = experiment_full(means, f, 5000, runs, seed, bern);
			record(csv, "E3_gap", "bernoulli", name, "gap", gap, means, 5000, runs, e);
		}
	}

	// ---------- E4：只改 eps ----------
	for(double eps : {0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 1.0}) {
		Solver_Factory f = [eps](int K) { return std::make_unique<Epsilon_Greedy>(K, eps); };
		const auto e = experiment_full(means_bernoulli, f, 2000, runs, seed, bern);
		record(csv, "E4_epsilon", "bernoulli", "eps-greedy", "eps", eps, means_bernoulli, 2000, runs, e);
	}

	// ---------- E5：只改 UCB 的 c ----------
	for(double c : {0.1, 0.2, 0.3, 0.5, 0.75, 1.0, std::sqrt(2.0), 2.0, 4.0}) {
		Solver_Factory f = [c](int K) { return std::make_unique<UCB>(K, c); };
		const auto e = experiment_full(means_bernoulli, f, 2000, runs, seed, bern);
		record(csv, "E5_ucb_c", "bernoulli", "UCB", "c", c, means_bernoulli, 2000, runs, e);
	}

	std::cout << "CSV 写入完成：results/curves.csv, results/summary.csv, results/counts.csv\n";

	// ---------- 校验：R_T = sum_i Delta_i N_i(T) ----------
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