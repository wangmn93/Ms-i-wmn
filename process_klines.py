import pandas as pd
import numpy as np
def generate_klines():
    return

def alpha_test_gen(df):
    # print('alpha test gen')
    # print('#fetched rows',len(fetched_rows))
    # print(df)
    return 0

def create_kline(klines, customer_vars=None):

    start_kline = klines.iloc[0]
    end_kline = klines.iloc[-1]
    kline_ = {'time':start_kline.time, 'date':start_kline.date, 'code':start_kline.code,'end_date':end_kline.date,
            'end_time':end_kline.time, 'period':klines.index.size}

    if customer_vars is not None:
        for var in customer_vars:
            kline_[var['name']] = var['function'](kline_)

    return kline_

#==============preprocess================
# adjust
def adjust_m1_klines(klines_df, adj_df, codes):
    results = []
    for code in codes:
        df = klines_df[klines_df['code']==code]
        dates = df['date'].unique().values
        for date in dates:
            df_ = df[df['date']==date]
            factor = adj_df[(adj_df['code']==code) & (adj_df['date']==date)]
            print(factor)
            # adjust price and volume
            results.append(df_)
   results_df = pd.concat(results)
   results_df.to_csv('adj_m1_klines.csv')
    
def create_new_klines(klines_df, codes, period, custom_vars=None):
    new_klines_all = []
    for code in codes:
        print('Creating period %d klines for code %d ...'%(period,code))
        df = klines_df[klines_df['code']==code]
        new_klines = []
        for idx in range(0, df.index.size):
            # if idx>=period:
            if df.index.size-idx >= period:
                fetched_rows = df.iloc[idx:idx+period]

                # print(fetched_rows)
                kline = create_kline(fetched_rows, custom_vars)
                new_klines.append(kline)
            else:
                print('#Rows is not enough, code %d, idx %d #total idx %d'%(code, idx, df.index.size))
        print('Done. Code',code,'period',period,'#num new klines', len(new_klines))
        new_klines_all += new_klines
        # print('Done...')
    new_klines_df = pd.DataFrame(new_klines_all)
    new_klines_df.to_csv('klines_period%d.csv'%period)
    print('Save klines to klines_period%d.csv'%period)

#============fast fetch rows================
# code = 6
# create row mask
# period=5
def generate_mask(period, num):
    mask = [i*(-period) for i in range(num)][::-1]
    return np.array(mask)

def compute_factor(klines_df,codes,period,generators):
    # klines_df = self.klines_df
    factors_list_all = []

    for code in codes:
        df = klines_df[klines_df['code']==code]
        # print(df)
        df_array = df.values

        # TODO create column idx dict
        column_idx = {}

        factors_list = []
        for idx in range(0, df.index.size):
            kline = df.iloc[idx]

            # TODO handle variable mask!!!
            time_idx = time2idx[kline['end_time']]

            factors = {'code':code, 'time':kline['end_time'],'date':kline['end_date']}
            for gen in factor_generators:
                name = gen['name']
                row_range = gen['range']
                # handle vatiable mask
                row_mask = gen['mask'] if gen['mask'] is not None else generate_mask(period, time_idx+1)
                func = gen['function']
                if idx>= row_range:
                    row_mask_ = row_mask + idx
                    # print(row_mask_)
                    fetch_rows = df.iloc[row_mask_]
                    # fetch_rows = df_array[row_mask_] # faster??? but need column idx dict
                    # print('#fetched rows',len(fetched_rows))
                    # print(feteched_rows)
                    # TODO call factors generators, pass rows and column index
                    factor = func(fetch_rows)
                    factors[name] = factor
                else:
                    factors[name] = np.nan
                    print('#Rows is not enough for factor %s, code %d, idx %d'%(name, code, idx))
            factors_list.append(factors)
        factors_list_all = factors_list_all + factors_list
    factors_all_df = pd.DataFrame(factors_list_all)
    factors_all_df.to_csv('factors_period%d.csv'%period)
#
#====================SMA=====================
def SMA(c, p, n, m):
    # a current
    # b previous
    return (float(c)*m + float(p)*n)/n

def compute_SMA(factors_df, codes, smas, func):
    results = []
    for code in codes:
        df = factors_df[factors_df['code']==code]
        smas_temp = [{'name':s['name'], 'n':s['n'],'m':s['m'],'column':df[s['factor']],'list':[]}for s in smas]
        for idx in range(0,df.index.size):
            for sma in smas_temp:
                if idx == 0:
                # handle first element
                    sma['list'].append(sma['column'].iloc[idx])
                else:
                # compute sma
                    y = SMA(c=sma['column'].iloc[idx], p=sma['list'][-1], n=sma['n'], m=sma['m'])
                    sma['list'].append(y)

        # call custom function
        sma_list = func(smas_temp)
        for s in sma_list:
            name = s['name']
            series = s['list']
            df[name] =  series
        results.append(df)
    results_df = pd.concat(results)
    print(results_df)
    print('Save to factor_sma.csv')
    results_df.to_csv('factors_sma.csv')

def var_test(kline):
    # return float(kline['high'])/kline['low'] - 1
    return -1.

if __name__ == '__main__':

    # test_df = [{'time': 930, 'date': 20180307, 'code': 6},
    #            {'time': 931, 'date': 20180307, 'code': 6},
    #            {'time': 932, 'date': 20180307, 'code': 6},
    #            {'time': 933, 'date': 20180307, 'code': 6},
    #            {'time': 934, 'date': 20180307, 'code': 6},
    #            {'time': 935, 'date': 20180307, 'code': 6},
    #            {'time': 936, 'date': 20180307, 'code': 6},
    #            {'time': 937, 'date': 20180307, 'code': 6},
    #            {'time': 938, 'date': 20180307, 'code': 6},
    #            {'time': 939, 'date': 20180307, 'code': 6},
    #            {'time': 930, 'date': 20180307, 'code': 20},
    #            {'time': 931, 'date': 20180307, 'code': 20},
    #            {'time': 932, 'date': 20180307, 'code': 20},
    #            {'time': 933, 'date': 20180307, 'code': 20},
    #            {'time': 934, 'date': 20180307, 'code': 20},
    #            {'time': 935, 'date': 20180307, 'code': 20},
    #            {'time': 936, 'date': 20180307, 'code': 20},
    #            {'time': 937, 'date': 20180307, 'code': 20},
    #            {'time': 938, 'date': 20180307, 'code': 20},
    #            {'time': 939, 'date': 20180307, 'code': 20},
    #            ]
    # test_df = pd.DataFrame(test_df)
    # test_df.to_csv('klines_test.csv')
    # for test
    '''
    Adjust 1m klines
    '''
    codes = [6, 20]
    klines_df = pd.read_csv('klines_test.csv')
    adj_df = pd.read_csv('adj_factor.csv')
    adjust_m1_klines(klines_df = klines_df, adj_df = adj_df, codes = codes)
    '''
    Preprocess klines
    '''
    codes = [6, 20]
    period = 1
    # test preprocess klines
    klines_df = pd.read_csv('klines_test.csv')
    custom_vars = [{'name':'ret', 'function':var_test}]
    create_new_klines(klines_df=klines_df, codes=codes, period=period, custom_vars=custom_vars)

    '''
    Compute factors
    '''
    new_klines_df = pd.read_csv('klines_period5.csv')
    # period should be consistent with klines period
    period = 5
    #factor args
    factor_args = [{'name':'alpha_test', 'period':period, 'num':2, 'function': alpha_test_gen},
                   {'name': 'alpha_test2', 'period': period, 'num': 1, 'function': alpha_test_gen}]
    factor_generators = []
    # convert args to row mask
    for args in factor_args:
        row_mask = generate_mask(period=period, num=args['num'])
        row_range = row_mask[-1] - row_mask[0]  # check this !!!
        factor_generators.append({'name':args['name'], 'mask':np.array(row_mask), 'range':row_range,
                                 'function':args['function']})
    print(factor_generators)
    # compute_factor(klines_df=new_klines_df, codes=codes, generators=factor_generators)


    '''
    Compute sma for factors 
    '''
    sma_args = [{'name':'sma1', 'factor':'alpha_test2', 'n':6, 'm':4},
            {'name': 'sma2', 'factor': 'alpha_test2', 'n': 42, 'm': 2}]
    factors_df = pd.read_csv('factors_period5.csv')
    def func_after_sma(smas):
        return smas

    # compute_SMA(factors_df=factors_df, codes=codes, smas=sma_args,
    #             func=custom_after_sma)

    
