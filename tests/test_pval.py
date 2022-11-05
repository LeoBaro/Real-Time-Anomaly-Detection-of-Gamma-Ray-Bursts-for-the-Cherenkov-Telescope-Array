from rtapipe.lib.evaluation.pval import *

class TestPval:

    def test_get_pvalue_from_prob(self):
        prob = 0.9973 # 2-tails probability to have a measure inside -3,+3 sigma
        # We are interested in:
        # the probability to have a measure outside -3,+3 sigma
        # we are interested in only one tail
        pvalue = get_pval_from_prob(prob, decimals=8) 
        assert  pvalue == 0.00135

        # this pvalue corresponds to 
        assert get_prob_from_pvalue(pvalue, decimals=10) == 0.9973


    def test_get_pvalue_from_sigma(self):
        sigma = 3.0
        pvalue = get_pvalue_from_sigma(sigma, decimals=10)
        assert pvalue == 0.0013498981

        # this pvalue corresponds to
        assert get_sigma_from_pvalue(pvalue, decimals=3) == 3.0

        # this sigma corresponds to probability
        assert get_prob_from_sigma(sigma, decimals=10) == 0.9973002039


        sigma = 5.0
        pvalue = get_pvalue_from_sigma(sigma, decimals=10)
        assert pvalue == 2.867e-07

        # this pvalue corresponds to
        assert get_sigma_from_pvalue(pvalue, decimals=3) == 5.0

        # this sigma corresponds to probability
        assert get_prob_from_sigma(sigma, decimals=10) == 0.9999994267

"""
sigma = 4
prob = 0.9973
pval = np.round((1-prob)/2, 10)

print(f'1-Tail probability of {sigma} sigma = {get_prob_from_sigma(sigma)}')
print(f'1-Tail probability of {pval} pvalue = {get_prob_from_pvalue(pval)}')
print(f'{pval} pvalue = {get_sigma_from_pvalue(pval)} sigma')
print(f'{sigma} sigma = {get_pvalue_from_sigma(sigma)} pvalue')
"""