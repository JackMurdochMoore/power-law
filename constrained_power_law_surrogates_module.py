import numpy as np# For pointwise and pairwise array operations
np.seterr(over='raise')# Make error when a floating point overflow is detected
import networkx as nx# For discrete zeta random variable
from sympy.ntheory import factorint# For prime factorisation
import random# For random number generation
from scipy.special import zeta
import math# For logarithms of large numbers
import operator# To check whether or not prime factor dictionaries are equal
#
#
# Adapted (slightly) from
# https://pythonhosted.org/combalg-py/_modules/combalg/subset.html#random
# (More randomisation added)
def random_k_element(elements, k):
    '''
    Returns a random k-element subset of a set of elements.  This is a clever algorithm, explained in [NW1978]_
    but not actually implemented.  Essentially starting with the first element of the universal set, include
    it in the subset with p=k/n.  If the first was selected, select the next with p=(k-1)/(n-1), if not
    p = k/(n-1), etc.

    :param elements: the input set
    :type elements: list
    :param k: the size of the output set
    :type k: int
    :return: a random k-element subset of the input set
    :rtype: list
    '''
    a = []
    c1 = k
    c2 = len(elements)
    i = 0
    while c1 > 0:
#         random.seed(timer())
#         if random.random() <= float(c1)/c2:
        if random.random() <= float(c1)/c2:
            a.append(elements[i])
            c1 -= 1
        c2 -= 1
        i += 1
    return a
# 
# Adapted (slightly) from
# https://pythonhosted.org/combalg-py/_modules/combalg/subset.html#random_k_element
# (More randomisation added)
def rand_comp(n,k):
    '''
    Returns a random composition of n into k parts.

    :param n: integer to compose
    :type n: int
    :param k: number of parts to compose
    :type k: int
    :return: a list of k-elements which sum to n

    Returns random element of :func:`compositions` selected
    `uniformly at random <https://en.wikipedia.org/wiki/Discrete_uniform_distribution>`_.
    '''
    if (k > 1):
#         random.seed(timer())
        a = random_k_element(range(1, n + k), k - 1)
        r = [0]*k
        r[0] = a[0] - 1
        for j in range(1,k-1):
            r[j] = a[j] - a [j-1] - 1
        r[k-1] = n + k - 1 - a[k-2]
        return r
    elif (k == 1):
        return [n]
    else:
        return []
# 
# 
def estim_scale_exp(seq, x_min):# Code for determining maximum likelihood scaling exponent scale_exp:
    scale_exp_list = np.arange(1.001, 10.000, 0.001);
    zeta_h_list = [zeta(scale_exp, x_min) for scale_exp in scale_exp_list]
    N = len(seq)
    log_deg_list = [math.log(deg) for deg in seq]
    log_deg_array = np.array(log_deg_list)
    sum_log_array = np.sum(log_deg_array)
    scale_exp_array = np.array(scale_exp_list)
    zeta_h_array = np.array(zeta_h_list)
    log_zeta_h_array = np.log(zeta_h_array)
    neg_log_like_array = np.add(np.multiply(scale_exp_array, sum_log_array), np.multiply(N, log_zeta_h_array))
    m_l_ind = np.argmin(neg_log_like_array)
    scale_exp_m_l = scale_exp_list[m_l_ind]
    return scale_exp_m_l
# 
# 
# Helper functions for producing power-law (and more general) surrogates:
# 
def subsample(seq, N=0):#Resampling by sampling without replacement (and replicating where necessary)
    N0 = len(seq)
    if (N == 0):
        N = N0
    if (N > N0):
        n = N//N0
        r = N%N0
        new_seq = []
        for ii in range(n):
            new_seq = new_seq + seq
        new_seq = new_seq + subsample(seq, N=r)
    elif (N == N0):
        new_seq = seq
    else:
        new_seq = random.sample(seq, k=N)
    random.shuffle(new_seq)
    return new_seq
# 
def bootstrap(seq, N=0):#Resampling by bootstrapping
    N0 = len(seq)
    if (N == 0):
        N = N0
    return random.choices(seq, k=N)
#
def resample(seq, method='subs', N=0):#Resampling by either: (1) bootstrapping, or (2) sampling without replacement (and replicating where necessary)
    if (method == 'boot'):
        return bootstrap(seq, N)
    else:
        return subsample(seq, N)        
#
def gen_typ_p_l_surrogate(scale_exp_m_l, x_min, N):# Generate from a power-law sequence with known parameters
    surr_seq = [int(nx.utils.zipf_rv(alpha=scale_exp_m_l, xmin=x_min)) for i in range(N)];
    return surr_seq
#
def build_factor_count_dict(seq):# Build a prime factor dictionary to assist with making constrained power-law surrogates:
    factor_count_dict = {}
    for deg in seq:
        deg_factor_count_dict = factorint(deg)
        for factor in deg_factor_count_dict:
            if not factor in factor_count_dict:
                factor_count_dict[factor] = deg_factor_count_dict[factor]
            else:
                factor_count_dict[factor] += deg_factor_count_dict[factor]
    return factor_count_dict
#
def make_move_factor_dict(obs_factor_dict, seq, x_min):#Work out which prime facors can be moved when making constrained power-law surrogates:
    N = len(seq)
    move_factor_dict = obs_factor_dict.copy()
    factor_list, _ = zip(*move_factor_dict.items())
    factor_list = list(factor_list)
#                                 factor_list.sort()
    factor_list.sort(reverse=True)
#                                 syst_rand_gen.shuffle(factor_list)
    num_dist_factors = len(factor_list)
    min_admiss_factor_ind_list = np.zeros(N).astype(int)
    val_fix_list = [0]*N
    for ii in range(N):
        deg = seq[ii]
        factorint_deg = factorint(deg)
        bin_count_list = [factorint_deg.get(factor) for factor in factor_list]
        bin_count_list = [0 if count is None else count for count in bin_count_list]
        current_factor_ind = next((i for i, x in enumerate(bin_count_list) if x), None)
        current_factor = factor_list[current_factor_ind]
        min_admiss_factor_ind_list[ii] = current_factor_ind
        val_fix = current_factor
        bin_count_list[current_factor_ind] -= 1
        move_factor_dict[current_factor] -= 1
        while val_fix < x_min:
            current_factor_ind = next((i for i, x in enumerate(bin_count_list) if x), None)
            current_factor = factor_list[current_factor_ind]
            min_admiss_factor_ind_list[ii] = current_factor_ind
            val_fix = val_fix*current_factor
            bin_count_list[current_factor_ind] -= 1
            move_factor_dict[current_factor] -= 1
        val_fix_list[ii] = val_fix
        min_admiss_factor_ind = next((i for i, x in enumerate(bin_count_list) if x), None)
    return (move_factor_dict, factor_list, min_admiss_factor_ind_list, val_fix_list)
# 
def gen_cons_power_law_surr_1(seq, prime_factor_count_dict):#Generate constrained power-law surrogate with x_min = 1:
    len_seq = len(seq)
    surr_seq = np.ones(len_seq, dtype = np.int64)
    prime_factor_count_list = list(prime_factor_count_dict.items())
    for factor, count in prime_factor_count_list:
        power_list = rand_comp(count, len_seq)
        surr_seq = np.multiply(surr_seq, np.power(factor, power_list))
    surr_seq = surr_seq.tolist()
    return surr_seq
#
def gen_cons_power_law_surr_2(move_factor_dict, factor_list, min_admiss_factor_ind_list, val_fix_list):#Generate constrained power-law surrogate with arbitrary positive integer x_min:
    len_seq = len(val_fix_list)
    surr_seq = val_fix_list
    prime_factor_count_list = list(move_factor_dict.items())
    for factor_ind in range(len(factor_list)):
        factor = factor_list[factor_ind]
        count = move_factor_dict[factor]
        admitt_seq_ind = [ii_ind for ii_ind in range(len_seq) if (factor_ind >= min_admiss_factor_ind_list[ii_ind])]
        len_admitt_seq = len(admitt_seq_ind)
        power_list_admitt_deg = rand_comp(count, len_admitt_seq)
        power_list_full = np.zeros(len_seq, dtype = np.int64)
        for jj_admitt in range(len_admitt_seq):
            power_list_full[admitt_seq_ind[jj_admitt]] = power_list_admitt_deg[jj_admitt]
        surr_seq = np.multiply(surr_seq, np.power(factor, power_list_full))
    surr_seq = surr_seq.tolist()
    return surr_seq
#
#
def gen_power_law_surr_list(seq, surr_method='cons', N=0, resamp_method='subs', x_min=1, num_surr=1, scale_exp=2.5):#Generate a list of power-law surrogates of the chosen type
    
    N0 = len(seq)
    
    if (N == 0):
        N = N0
        
    if (surr_method == 'know'):#Known scale exponent
        surr_list = [gen_typ_p_l_surrogate(scale_exp, x_min, N) for ii in range(num_surr)]
    
    elif (surr_method == 'boot'):#Bootstrap
        surr_list = [bootstrap(seq, N) for ii in range(num_surr)]
    
    elif (surr_method == 'subs'):#Up- or down-sample
        if (N == N0):
            surr_list = [seq for ii in range(num_surr)]
        else:
            surr_list = [subsample(seq, N) for ii in range(num_surr)]
    
    elif (surr_method == 'typi'):#Typical surrogates
        if ((N == N0) & (resamp_method == 'subs')):
            m_l_scale_exp = estim_scale_exp(seq, x_min)#Maximum likelihood scale exponent
            surr_list = [gen_typ_p_l_surrogate(m_l_scale_exp, x_min, N) for ii in range(num_surr)]
        else:
            surr_list = []
            for ii in range(num_surr):
                resamp_seq = resample(seq, method=resamp_method, N=N)
                m_l_scale_exp = estim_scale_exp(resamp_seq, x_min)#Maximum likelihood scale exponent
                surr_list = surr_list + [gen_typ_p_l_surrogate(m_l_scale_exp, x_min, N) for ii in range(num_surr)]
    
    else:#Constrained surrogate
        surr_list = []
        if ((N == N0) & (resamp_method == 'subs')):#Do not need to resample
            obs_factor_dict = build_factor_count_dict(seq)
            if (x_min == 1):
                for ii in range(num_surr):
                    surr_seq = gen_cons_power_law_surr_1(seq, obs_factor_dict)
                    surr_factor_dict = build_factor_count_dict(surr_seq)
                    while not operator.eq(obs_factor_dict, surr_factor_dict):
                        set1 = set(obs_factor_dict.items())
                        set2 = set(surr_factor_dict.items())
                        print('There is a problem: prime factor counts of surrogate do not match those of original. Difference:')
                        print(set1 ^ set2)
                        print('We will generate a replacement surrogate.')
                        surr_seq = gen_cons_power_law_surr_1(seq, obs_factor_dict)
                        surr_factor_dict = build_factor_count_dict(surr_seq)
                    surr_list = surr_list + [surr_seq]
            else:
                (move_factor_dict, factor_list, min_admiss_factor_ind_list, val_fix_list) = make_move_factor_dict(obs_factor_dict, seq, x_min)
                for ii in range(num_surr):
                    surr_seq = gen_cons_power_law_surr_2(move_factor_dict, factor_list, min_admiss_factor_ind_list, val_fix_list)
                    surr_factor_dict = build_factor_count_dict(surr_seq)
                    while not operator.eq(obs_factor_dict, surr_factor_dict):
                        set1 = set(obs_factor_dict.items())
                        set2 = set(surr_factor_dict.items())
                        print('There is a problem: prime factor counts of surrogate do not match those of original. Difference:')
                        print(set1 ^ set2)
                        print('We will generate a replacement surrogate.')
                        surr_seq = gen_cons_power_law_surr_2(move_factor_dict, factor_list, min_admiss_factor_ind_list, val_fix_list)
                        surr_factor_dict = build_factor_count_dict(surr_seq)
                    surr_list = surr_list + [surr_seq]
        else:#Do need to resample
            if (x_min == 1):
                for ii in range(num_surr):
                    valid_surr_flag = False
                    while not valid_surr_flag:
                        resamp_seq = resample(seq, method=resamp_method, N=N)
                        resamp_factor_dict = build_factor_count_dict(resamp_seq)
                        surr_seq = gen_cons_power_law_surr_1(resamp_seq, resamp_factor_dict)
                        surr_factor_dict = build_factor_count_dict(surr_seq)
                        if not operator.eq(resamp_factor_dict, surr_factor_dict):
                            set1 = set(resamp_factor_dict.items())
                            set2 = set(surr_factor_dict.items())
                            print('There is a problem: prime factor counts of surrogate do not match those of bootstrapped. Difference:')
                            print(set1 ^ set2)
                            print('We will generate a replacement surrogate.')
                            valid_surr_flag = False
                        else:
                            valid_surr_flag = True
                    surr_list = surr_list + [surr_seq]
            else:
                for ii in range(num_surr):
                    valid_surr_flag = False
                    while not valid_surr_flag:
                        resamp_seq = resample(seq, method=resamp_method, N=N)
                        resamp_factor_dict = build_factor_count_dict(resamp_seq)
                        (move_factor_dict, factor_list, min_admiss_factor_ind_list, val_fix_list) = make_move_factor_dict(resamp_factor_dict, resamp_seq, x_min)
                        surr_seq = gen_cons_power_law_surr_2(move_factor_dict, factor_list, min_admiss_factor_ind_list, val_fix_list)
                        surr_factor_dict = build_factor_count_dict(surr_seq)
                        while not operator.eq(resamp_factor_dict, surr_factor_dict):
                            set1 = set(obs_factor_dict.items())
                            set2 = set(surr_factor_dict.items())
                            print('There is a problem: prime factor counts of surrogate do not match those of bootstrapped. Difference:')
                            print(set1 ^ set2)
                            print('We will generate a replacement surrogate.')
                            valid_surr_flag = False
                        else:
                            valid_surr_flag = True
                    surr_list = surr_list + [surr_seq]
    
    return surr_list