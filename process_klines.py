import pandas as pd
#==============preprocess================
#for test
codes = [6]
period = 5
klines_df = self.klines_df
new_klines_all = []
for code in codes:
  print('Creating period %d klines for code %d ...'%(period,code))
  df = klines_df[klines_df==code]:
  new_klines = []
  for idx in range(0, df_.index.size):
      if idx>=period:
        fetched_rows = df.iloc[[idx-period:idx]]
        kline = self.create_kline(fetched_rows)
        new_klines.append(kline)
      else:
        print('#Previous rows is not enough, code %d, idx %d',%(code, idx))
  print('code',code,'period',period,'#num new klines', len(new_klines))
  new_klines_all += new_kline
  print('Done...')
new_klines_df = pd.DataFrame(new_klines_all)
new_klines_df.to_csv('klines_period%d.csv'%period)

#============fast fetch rows================
# code = 6
# create row mask 
period=5
num = 57
row_mask = [i*(-period) for i in range(num)][::-1]
row_range = row_mask[-1] - row_mask[0] # check this !!!

# TODO bind factors generators, row mask and row range
factor_generators [{'name':,'mask','function'}]
# read new df first !!!
klines_df = self.klines_df
factors_list_all = []

for code in codes:
  df = klines_df[klines_df==code]
  df_array = df.values

  # TODO create column idx dict
  column_idx = {}
  factors_list = []
  for idx in range(0, df_.index)
    factors = {'code':code, 'time':None,'date':None}
    for gen in factor_generators:
      name = gen['name']
      row_range = gen['range']
      row_mask = gen['mask']
      if idx>= row_range:
        row_mask_ = row_mask + idx
        fetch_rows = df.iloc[row_mask_]
        # fetch_rows = df_array[row_mask_] # faster
        print('#fetched rows',len(fetched_rows))
        print(feteched_rows)
        # TODO call factors generators, pass rows and column index
        factor = None
        factors[name] = factor
       else:
         print('#Previous row is not enough for factor %s, code %d, idx %d'%(name, code, idx))
     factors_list.append(factors)
   
  factor_list_all += factors_list
 factors_df = pd.DateFrame(factor_list_all)
 factor_df.to_csv('factors_period%d.csv'%period)
 
 #====================SMA=====================
 factors_df = None # read factors
 smas = [{'name':,'n','m'},
 {'name':,'n','m'},
 smas
 for code in codes:
  column_df = factors_df[factors_df==code]['factor name']
  
  sma_list = {'fname':[], 'fname2':[]}
  for idx in range(0:column_df.index.size):
    for sma in smas:
      n_ = sma['n']
      m_ = sma['m']
      if idx == 0:
        pass
      else:
        sma.append
      
  
  


    
