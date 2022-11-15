#ifndef __PROBLEM_H__
#define __PROBLEM_H__

#include <cstddef>
#include <eigen3/Eigen/Core>
#include <iostream>

static const int kRosenbrockN = 2;

namespace optimization_solver {
enum ProblemType {
  Example1,
  Example2,
  Example3,
  Example4,
  PRosenbrock,
};

class Problem {
public:
  virtual double GetCost(const Eigen::VectorXd &x) = 0;
  virtual Eigen::VectorXd GetGradient(const Eigen::VectorXd &x);
  virtual Eigen::MatrixXd GetHessian(const Eigen::VectorXd &x);
  virtual Eigen::VectorXd GetDiffGradient(const Eigen::VectorXd &x) final;
  virtual Eigen::MatrixXd GetDiffHessian(const Eigen::VectorXd &x) final;
  virtual std::size_t GetProblemSize() final { return size_; }

protected:
  int size_ = 0;
};

class RosenbrockFunction : public Problem {
public:
  RosenbrockFunction() { size_ = kRosenbrockN; }
  double GetCost(const Eigen::VectorXd &x) override;
  Eigen::VectorXd GetGradient(const Eigen::VectorXd &x) override;
  Eigen::MatrixXd GetHessian(const Eigen::VectorXd &x) override;

private:
};

class Example1Func : public Problem {
public:
  Example1Func() { size_ = 2; }
  double GetCost(const Eigen::VectorXd &x) override;
  // Eigen::VectorXd GetGradient(const Eigen::VectorXd &x) override;
  // Eigen::MatrixXd GetHessian(const Eigen::VectorXd &x) override;

private:
};

class Example2Func : public Problem {
public:
  Example2Func() { size_ = 2; }
  double GetCost(const Eigen::VectorXd &x) override;
  // Eigen::VectorXd GetGradient(const Eigen::VectorXd &x) override;
  // Eigen::MatrixXd GetHessian(const Eigen::VectorXd &x) override;

private:
};

class Example3Func : public Problem {
public:
  Example3Func() { size_ = 2; }
  double GetCost(const Eigen::VectorXd &x) override;
  Eigen::VectorXd GetGradient(const Eigen::VectorXd &x) override;
  // Eigen::MatrixXd GetHessian(const Eigen::VectorXd &x) override;

private:
};

class Example4Func : public Problem {
public:
  Example4Func() { size_ = 2; }
  double GetCost(const Eigen::VectorXd &x) override;
  // Eigen::VectorXd GetGradient(const Eigen::VectorXd &x) override;
  // Eigen::MatrixXd GetHessian(const Eigen::VectorXd &x) override;

private:
};

}; // namespace optimization_solver

#endif