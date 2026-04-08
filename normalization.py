
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取 CSV 文件
df = pd.read_csv('LSTM.csv')

# 分离特征和目标列
features = df.drop(columns=['Q_obs'])
target = df['Q_obs']

# 仅对数值型特征列进行归一化
numeric_features = features.select_dtypes(include=['number'])

# 记录非数值列
non_numeric = features.select_dtypes(exclude=['number'])

# 归一化数值列
scaler = MinMaxScaler()
normalized_numeric = pd.DataFrame(scaler.fit_transform(numeric_features), columns=numeric_features.columns)

# 合并归一化后的数值列、非数值列（如果有），以及目标列
df_processed = pd.concat([normalized_numeric, non_numeric.reset_index(drop=True), target.reset_index(drop=True)], axis=1)

# 保存处理后的数据（可选）
df_processed.to_csv('LSTM_normalized2.csv', index=False)

# 打印前几行确认
print(df_processed.head())
'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取 CSV 文件
df = pd.read_csv('LSTM_with_Ep_AI_h_Sproxy.csv')

# 不参与归一化的列
exclude_cols = ['date', 'Q_obs']

# 分离目标列和日期列
target = df['Q_obs']
date = df['date']

# 选取需要归一化的数值型特征列（排除 exclude_cols）
features_to_normalize = df.drop(columns=exclude_cols)
numeric_cols = features_to_normalize.select_dtypes(include=['number']).columns

# 归一化
scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(features_to_normalize[numeric_cols])
normalized_df = pd.DataFrame(normalized_values, columns=numeric_cols)

# 拼接成最终 DataFrame，保持原始顺序：date、归一化特征、Q_obs
df_processed = pd.concat([date.reset_index(drop=True),
                          normalized_df.reset_index(drop=True),
                          target.reset_index(drop=True)],
                         axis=1)

# 保存处理结果
df_processed.to_csv('LSTM_normalized.csv', index=False)

# 打印确认
print(df_processed.head())
'''