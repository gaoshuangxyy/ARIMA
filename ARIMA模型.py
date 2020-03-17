import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

Data = pd.read_excel('data.xlsx',index_col = 'date',parse_dates=['date'])
#划分训练集和测试集
sub = Data['2012-01':'2018-09']['H2'].resample('M').mean()
train = sub['2012-01':'2016-12']
test = sub['2016-11':'2018-09']

#二阶差分
Data['H2_diff_1'] = Data['H2'].diff(1)
Data['H2_diff_2'] = Data['H2_diff_1'].diff(1)
plt.figure(figsize=(20,6))

plt.subplot(131)
plt.title('H2')
plt.plot(Data['H2'])

plt.subplot(132)
plt.title('diff_1')
plt.plot(Data['H2_diff_1'])

plt.subplot(133)
plt.title('diff_2')
plt.plot(Data['H2_diff_2'])

#画出acf和pacf图
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train, lags=50, ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
# fig.tight_layout()
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train, lags=50, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout()
# #热度图
# import itertools
# import numpy as np
# import seaborn as sns
#
# p_min = 0
# d_min = 0
# q_min = 0
# p_max = 5
# d_max = 0
# q_max = 5
#
# # Initialize a DataFrame to store the results,，以BIC准则
# results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
#                            columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])
#
# for p, d, q in itertools.product(range(p_min, p_max + 1),
#                                  range(d_min, d_max + 1),
#                                  range(q_min, q_max + 1)):
#     if p == 0 and d == 0 and q == 0:
#         results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
#         continue
#
#     try:
#         model = sm.tsa.ARIMA(train, order=(p, d, q),
#                              # enforce_stationarity=False,
#                              # enforce_invertibility=False,
#                              )
#         results = model.fit()
#         results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
#     except:
#         continue
# results_bic = results_bic[results_bic.columns].astype(float)
#
# fig, ax = plt.subplots(figsize=(10, 8))
# ax = sns.heatmap(results_bic,
#                  mask=results_bic.isnull(),
#                  ax=ax,
#                  annot=True,
#                  fmt='.2f',
#                  )
# ax.set_title('BIC')
# plt.show()
#
# train_results = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='nc', max_ar=8, max_ma=8)
# print('AIC', train_results.aic_min_order)
# print('BIC', train_results.bic_min_order)
#
# # 模型残差检验
# model = sm.tsa.ARIMA(train, order=(1, 0, 1))
# results = model.fit()
# resid = results.resid #赋值
# fig = plt.figure(figsize=(12,8))
# fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40)
# # plt.show()
#
# resid = results.resid#残差
# plt.figure(figsize=(12,8))
# ax = fig.add_subplot(111)
# plt.plot(resid, line='q', ax=ax, fit=True)

#模型预测
model = sm.tsa.ARIMA(sub, order=(1, 2, 0))
results = model.fit()
predict_sunspots = results.predict(start=str('2016-11'),end=str('2018-09'),dynamic=False)
print(predict_sunspots)
plt.figure(figsize=(12,8))
plt.plot(predict_sunspots,color = 'r',label = 'predict')
plt.plot(sub,color = 'b',label = 'original')
plt.legend(loc = 'best')
plt.show()

print('........')
print(sub['2016-11':'2018-09'])