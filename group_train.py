 import pandas as pd
    from misc import query_data
    import numpy as np
    from sklearn.model_selection import train_test_split
    from scipy.stats import pearsonr
    from sklearn.ensemble import RandomForestRegressor

    df = pd.read_hdf('data_insample.h5')
    r_list = []
    for i in range(3,4):
        X_groups = []
        y_groups = []
        test_groups = []
        selections = []
        for code in range(i*10, (i+1)*10):
        # for code in [0,2,6,7,8]:
        # for code in [10, 11, 15, 17, 18, 19]:
        # for code in [110,111,112,115,116,117,119]:
        #     print('=================')
            # print('query code', code)
            data = query_data(df, code, lag=0)
            if data is None:
                # print('Not enough data')
                pass
            else:
                lay_5_y = data.iloc[:, -1].shift(5)
                # lag_10_y = data.iloc[:, -1].shift(10)
                new = pd.concat(objs=[data.iloc[:, 0:-1], lay_5_y], axis=1, join='inner')[5:]
                # new = data.iloc[:, 0:-1][5:]
                # X = np.array(data.iloc[:, 0:-1])
                # Y = np.array(data.iloc[:, -1])
                X = np.array(new)
                Y = np.array(data.iloc[:, -1][5:])
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.22, shuffle=False)

                if X_train.shape[1] == 295:
                    print('#Train data', len(X_train), '#Test data', len(X_test), X_train.shape[1])
                    X_groups.append(X_train)
                    y_groups.append(y_train)
                    test_groups.append([code, X_test[:-100], y_test[:-100]])
                    selections.append(code)

        huge_X = np.concatenate(tuple(X_groups), axis=0)
        huge_y = np.concatenate(tuple(y_groups), axis=0)
        print('[Group %d]Total train points: %d'%(i, len(huge_X)))
        print('[Group %d]'%i, selections)
        # model = NeuralNetwork(tf_net, huge_X.shape[1])
        # model.fit(huge_X, huge_y, batchsize=50, epoch=50)
        model = RandomForestRegressor(n_estimators=500, max_depth=150, n_jobs=-1).fit(huge_X, huge_y)
        r_temp = []
        for c, t_x, t_y in test_groups:
            r = pearsonr(model.predict(t_x), t_y)
            print('[Group %d]Code %d'%(i,c),r)
            r_temp.append(r[0])
            r_list.append(r[0])
        print('=====>[Group %d]Average correlation: %.4f'%(i, np.mean(r_temp)))
        #     if code % 10 == 0:
        #         print('=====Avg correlation', np.mean(r_list))
    print('=====>Average correlation %.4f'%np.mean(r_list))
    print(r_list)
