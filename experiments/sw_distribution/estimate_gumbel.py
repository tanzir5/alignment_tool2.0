#TODO: Find a good way to estimate values of m, n
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from scipy.optimize import minimize
import pandas as pd
from scipy.optimize import LinearConstraint
import copy

XLIM = 4
m, n = 500, 500
global_data = None
def gaussian(params):
  mean = params[0]   
  sd = params[1]
  nll = -np.sum(stats.norm.logpdf(data, loc=mean, scale=sd))
  return nll

def get_stats(a):
  return np.min(a), np.median(a), np.max(a), np.mean(a), np.std(a)

def draw_normal_for_data(scores):
  mu = np.mean(scores)
  sigma = np.std(scores)
  print("mean/mu =", mu, "sigma/stddev =", sigma)
  x = np.linspace(0, XLIM, 100000)
  plt.xlim(left=0.0,right=XLIM)
  plt.plot(x, stats.norm.pdf(x, mu, sigma), "g-")
  

def draw_kstest(data, target_dist):
  data_sorted = np.sort(data)

  # calculate the proportional values of samples
  p = 1. * np.arange(len(data)) / (len(data) - 1)

  fig = plt.figure(figsize=(10,20))
  ax2 = fig.add_subplot(121)
  ax2.locator_params(nbins=15)
  ax2.plot(data_sorted, p)
  ax2.set_ylabel('$cdf$')
  ax2.plot(x, target_dist, "r-")


def gumbel_log_pdf(data, u, beta):
  ret = []
  log_inv_beta = np.log(1/beta)
  for x in data:
    z = (x - u) / beta
    ret.append(-(z+np.exp(-z)) + log_inv_beta)
  return np.array(ret)

def gumbel_pdf(data, u, beta):
  ret = []
  for x in data:
    z = (x - u) / beta
    y = (1/beta) * np.exp(-(z+np.exp(-z)))
    ret.append(y)
  return ret

def gumbel_cdf(data, u, beta):
  ret = []
  for x in data:
    y = np.exp(-np.exp(-(x-u)/beta))
    ret.append(y)
  return ret

def gumbel_loss(params):
  K = params[0]
  _lambda = params[1]
  u = np.log(K * m * n) / _lambda
  beta  = 1 / _lambda
  nll = -np.sum(gumbel_log_pdf(global_data, u, beta))
  return nll

def estimate_gumbel(data):
  initParams = [2, 1]
  #results = minimize(gumbel_loss, initParams, method='Nelder-Mead')
  linear_constraint = LinearConstraint(
    [[1, 0], [0, 1]], [0, 0], [np.inf, np.inf])
  results = minimize(gumbel_loss, initParams, method='trust-constr',
                     constraints=[linear_constraint])

  print("Gumbel estimation outputs:")
  print(results)
  print(results.success)
  print(results.message)
  K, _lambda = results.x[0], results.x[1]
  print("Gumbel K and lambda")
  print(K, _lambda)
  u = np.log(K * m * n) / _lambda
  beta = 1 / _lambda
  return u, beta

def gumbel_cdf_global(data):
  return gumbel_cdf(data, u, beta)

def draw_gumbel_over_data(data, u, beta):
  x = np.linspace(0, 40, 10000)
  ax = sns.displot(data,  stat='density')#ax=ax)
  data_sorted = np.sort(data)
  # calculate the proportional values of samples
  p = 1. * np.arange(len(data)) / (len(data) - 1)
  plt.xlim(0, XLIM)
  plt.plot(x, gumbel_pdf(x, u, beta), "r-", linewidth=1.5)
  plt.xlabel('Maximum SW Alignment Score')
  return

if __name__ == "__main__":
  #plt.rcParams.update({'font.size': 13.5})
  df = pd.read_csv('data/experiments/sw_distribution/good_sw_scores_sbert.csv')
  scores = np.array(df['score'])
  global_data = copy.deepcopy(scores)
  print("min, median, max, mean, std of scores are")
  print(get_stats(scores))
  print("0 scores fraction: ", np.sum(scores==0)/scores.size)
  #u, beta = estimate_gumbel(scores)
  u, beta = (1.2911376811296125, 0.3029363212261883)
  print("gumbel parameters", u, beta)
  #print(stats.kstest(scores, gumbel_cdf_global))
  draw_gumbel_over_data(scores, u, beta)
  #draw_normal_for_data(scores)
  #draw_kstest(scores)
  plt.show()

'''
Gumbel K and lambda
0.0002838286040066365 3.3010237793616932
gumbel parameters 1.2911376811296125 0.3029363212261883
'''