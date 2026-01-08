# coding=gbk
import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier  # 改为随机森林
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')



def knn_impute(df, label_col, n_neighbors=3, test_size=0.1):

    groups = df.groupby(label_col)

    imputer = KNNImputer(n_neighbors=n_neighbors)

    imputed_dfs = []
    for label, group in groups:
        for i, key in enumerate(group):
            if i != 0:
                if n_neighbors > 0:
                    try:
                        group[key] = imputer.fit_transform(np.array(group[key]).reshape(-1, 1))
                    except:
                        stop=1
                else:
                    group.loc[:, str(key)] = group.loc[:, str(key)].fillna(group.loc[:, str(key)].median())

        imputed_dfs.append(group)

    df = pd.concat(imputed_dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=43).reset_index(drop=True)

    return df

def replace_outliers_with_nan(data, threshold=1.5):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    data = np.where((data < lower_bound) | (data > upper_bound), np.nan, data)
    return data


if __name__ == '__main__':
    usehyparams_search = False
    # usehyparams_search = True
    useknn_impute = True
    file_name = 'Au_Mo_ratio_label.xlsx'

    df = pd.read_excel(file_name)

    columns_to_drop = ['La', 'Pr']
    df = df.drop(columns=columns_to_drop)

    label_encoder = LabelEncoder()
    df['label'] = df['label'].astype(str)
    df['label'] = label_encoder.fit_transform(df['label'])
    class_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print(class_mapping.keys())
    print("Class to Index Mapping:")
    for class_name, index in class_mapping.items():
        print(f"{class_name}: {index}")

    for col in df.columns:
        if col != 'label':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = replace_outliers_with_nan(df[col])

    nan_percentage = df.isna().mean() * 100
    threshold = 30

    columns_to_drop = nan_percentage[nan_percentage > threshold].index.tolist()

    df = df.drop(columns=columns_to_drop)

    df = df.dropna(how='all')

    if useknn_impute:
        # X_train, X_test, y_train, y_test = knn_impute(df, 'label')
        df = knn_impute(df, 'label')
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df['label'], test_size=0.1,
                                                          random_state=42)
        X, y = df.iloc[:, 1:], df['label']
    else:
        X, y = df.iloc[:, 1:], df['label']
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df['label'], test_size=0.1,
                                                            random_state=42)
    df.to_excel('data_process.xlsx', index=False)


    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 类别权重平衡
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = np.ones(len(X_train))
    for class_label, weight in enumerate(class_weights):
        sample_weights[y_train == class_label] = weight

    # 随机森林模型配置
    if not usehyparams_search:
        # 使用默认参数
        clf = RandomForestClassifier(
            n_estimators=1200,  # 搜索得到的最优值
            max_depth=16,  # 搜索得到的最优值
            min_samples_split=2,  # 搜索得到的最优值
            min_samples_leaf=1,  # 搜索得到的最优值
            max_features=0.4394481904373274,  # 搜索得到的最优值
            bootstrap=False,  # 搜索得到的最优值
            random_state=42,
            class_weight='balanced'
        )
    else:
        # 贝叶斯超参数搜索
        clf = RandomForestClassifier(random_state=42, class_weight='balanced')

        param_space = {
            'n_estimators': Integer(50, 1200),
            'max_depth': Integer(1, 20),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Real(0.1, 1.0),  # 特征采样比例
            'bootstrap': [True, False]
        }

        opt = BayesSearchCV(
            clf,
            param_space,
            n_iter=100,  # 可以增加迭代次数以获得更好结果
            cv=5,
            scoring='f1_macro',
            n_jobs=1,
            random_state=42,
            verbose=0
        )

        opt.fit(np.row_stack((X_train, X_test)), np.concatenate((y_train, y_test)))

        search_res = np.column_stack(
            (np.array(opt.optimizer_results_[0]['x_iters']), -1 * opt.optimizer_results_[0]['func_vals']))

        column_names = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features',
                        'bootstrap', "score"]
        search_res_df = pd.DataFrame(search_res, columns=column_names)
        search_res_df.to_csv(f'search_res_rf.csv', index=False)

        clf.set_params(**opt.best_params_)
        print('best-f1:', opt.best_score_)
        print('best-params:', opt.best_params_)

    # 交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average='macro')
    scores = cross_val_score(clf, X, y, cv=kf, scoring=f1_scorer)

    for i, score in enumerate(scores):
        print(f"Fold {i + 1} F1 Score: {score}")

    print(f"Mean F1 Score: {scores.mean()}")

    # 模型训练（随机森林不需要sample_weight参数，因为已经有class_weight）
    clf.fit(X_train, y_train)

    # 预测
    y_pred = clf.predict(X_test)

    # 特征重要性
    feature_importance = clf.feature_importances_

    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    # 保存模型和标准化器
    joblib.dump(scaler, 'fertility_scaler_rf.pkl')
    joblib.dump(clf, 'fertility_model_rf.pkl')

    # 重新读取原始数据用于特征名称
    df_original = pd.read_excel(file_name)
    columns_to_drop1 = ['La', 'Pr']
    df_original = df_original.drop(columns=columns_to_drop1)
    df_original = df_original.drop(columns=columns_to_drop)

    # 特征重要性可视化（随机森林内置）
    top_10_indices = np.argsort(feature_importance)[::-1][:10]  # 取前10个
    top_10_feature_names = [df_original.columns[1:][i] for i in top_10_indices]
    top_10_feature_importances = feature_importance[top_10_indices] / np.sum(feature_importance)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_10_feature_names, top_10_feature_importances, color='lightgreen')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Feature Name')
    plt.title('Top 10 Feature Importances in Random Forest Model')
    plt.gca().invert_yaxis()

    for bar, importance_score in zip(bars, top_10_feature_importances):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{importance_score:.3f}', ha='left', va='center')

    plt.savefig('feature_importance_rf.png', dpi=600)
    plt.show()

    # SHAP分析（需要为树模型使用TreeExplainer）
    explainer = shap.TreeExplainer(clf)  # 改为TreeExplainer
    shap_values = explainer.shap_values(X_train)
    np.save('shap_values_3d.npy', shap_values)
    shap_values = np.load('shap_values_3d.npy')
    print(shap_values)
    # # ---------------------------------------------------------
    # # 1. 准备特征名称 (确保与X_train的列对应)
    # # ---------------------------------------------------------
    # # 注意：你的X_train已经是numpy数组且经过了标准化，我们需要从df中提取对应的列名
    # # df在前面的步骤中已经剔除了NaN过多的列和Label列，所以除去label列即为特征
    feature_names = df.drop(columns=['label']).columns.tolist()

    plt.figure(figsize=(10, 8))
    plt.title(f"(Target Class: {label_encoder.inverse_transform([0])[0]})",
              fontsize=14)
    # plt.xlabel(f"SHAP value (Target Class: {label_encoder.inverse_transform([0])[0]})")
    # 绘制蜂群图
    # feature_names: 特征名称
    # X_train: 特征数值（用于决定点的颜色：红色高值，蓝色低值）
    shap.summary_plot(shap_values[:,:,0], X_train, feature_names=feature_names, plot_type="dot", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_dot_plot_A.png', dpi=600)  # 保存高清图
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.title(f"(Target Class: {label_encoder.inverse_transform([1])[0]})",
              fontsize=14)
    # plt.xlabel(f"SHAP value (Target Class: {label_encoder.inverse_transform([1])[0]})")
    # 绘制蜂群图
    # feature_names: 特征名称
    # X_train: 特征数值（用于决定点的颜色：红色高值，蓝色低值）
    shap.summary_plot(shap_values[:,:,1], X_train, feature_names=feature_names, plot_type="dot", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_dot_plot_B.png', dpi=600)  # 保存高清图
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.title(f"(Target Class: {label_encoder.inverse_transform([2])[0]})",
              fontsize=14)
    # plt.xlabel(f"SHAP value (Target Class: {label_encoder.inverse_transform([2])[0]})")
    # 绘制蜂群图
    # feature_names: 特征名称
    # X_train: 特征数值（用于决定点的颜色：红色高值，蓝色低值）
    shap.summary_plot(shap_values[:,:,2], X_train, feature_names=feature_names, plot_type="dot", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_dot_plot_C.png', dpi=600)  # 保存高清图
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.title(f"(Target Class: {label_encoder.inverse_transform([3])[0]})",
              fontsize=14)
    # plt.xlabel(f"SHAP value (Target Class: {label_encoder.inverse_transform([3])[0]})")
    # 绘制蜂群图
    # feature_names: 特征名称
    # X_train: 特征数值（用于决定点的颜色：红色高值，蓝色低值）
    shap.summary_plot(shap_values[:, :, 3], X_train, feature_names=feature_names, plot_type="dot", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_dot_plot_D.png', dpi=600)  # 保存高清图
    plt.show()
    #
    # # ---------------------------------------------------------
    # # 3. 绘制图 (B): Mean Absolute SHAP Value (条形图)
    # # ---------------------------------------------------------
    # # 想要复现图(B)那种堆叠条形图的效果（显示不同类别对模型的影响），
    # # 我们直接传入完整的 shap_values 列表，并将 plot_type 设置为 'bar'
    #
    plt.figure(figsize=(10, 8))
    plt.title("Mean Absolute SHAP Value by Class", fontsize=14)
    # plot_type="bar" 会自动计算平均绝对值
    # 传入整个列表 shap_values 可以看到多类别的堆叠效果 (类似于参考图B)
    # 如果只想看整体重要性不分颜色，可以传入 shap_values_target
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary_bar_plot_B.png', dpi=600)
    plt.show()


    # 混淆 矩阵
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 12})
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks(np.arange(len(np.unique(y_test))), class_mapping.keys())
    plt.yticks(np.arange(len(np.unique(y_test))), class_mapping.keys())
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    for i in range(len(np.unique(y_test))):
        for j in range(len(np.unique(y_test))):
            plt.text(j, i, f'{cm_percent[i, j]:.2f}', ha='center', va='center')
    plt.savefig('confusionmatrix_rf.png', dpi=600)
    plt.show()

    # --- 假设前面的数据准备代码保持不变 ---
    classes = np.unique(y)
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_score = clf.predict_proba(X_test)

    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # --- 绘图部分修改 ---

    # 1. 设置画布风格，使用更干净的背景
    plt.figure(figsize=(7, 6))  # 调整为稍微偏正方形的比例
    ax = plt.gca()  # 获取当前轴对象，方便后续操作边框

    # 2. 定义目标图片中的配色 (近似色值)
    # 对应：深蓝, 橙色, 绿色, 浅蓝, 红色
    custom_colors = ['#4c72b0', '#dd8452', '#55a868', '#8172b3', '#c44e52']
    # 如果类别超过5个，会自动循环；你也可以使用 seaborn 的配色
    colors = cycle(custom_colors)

    # 3. 绘制每个类别的ROC曲线
    for i, color in zip(range(n_classes), colors):
        # 获取类别名称，处理字典键值
        class_label = list(class_mapping.keys())[i]
        print(class_label)
        # 格式化图例标签，模仿目标图样式 "Name: AUC"
        # 注意：目标图中括号内的(0.90, 0.92)是置信区间，标准sklearn代码无法直接计算。
        # 这里我们只显示AUC值，格式保留两位小数
        label_text = f'Class {class_label} AUC:{roc_auc[i]:.3f}'

        plt.plot(fpr[i], tpr[i], color=color, lw=1, label=label_text)

    # 4. 绘制对角线 (目标图是浅灰色虚线)
    plt.plot([0, 1], [0, 1], color='lightgray', linestyle='--', lw=1.5)

    # 5. 设置坐标轴范围和标签 (修改为 Sensitivity 和 1-Specificity)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])  # 稍微留一点顶部空间
    plt.xlabel('1-Specificity', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)

    # 6. 核心样式调整：去边框和设置网格
    # 去掉上方和右方的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 加深左方和下方的边框颜色
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_position(('outward', 7))
    # 设置网格：只保留水平网格 (axis='y')，颜色浅灰
    plt.grid(axis='y', linestyle='-', alpha=0.3, color='lightgray')
    plt.grid(axis='x', visible=False)  # 关闭垂直网格

    # 7. 图例设置
    # 目标图的图例在右下角，背景半透明或白色，带边框
    legend = plt.legend(loc="lower right", frameon=True, fontsize=10,
                        edgecolor='lightgray', fancybox=True)
    legend.get_frame().set_alpha(0.8)  # 设置图例背景透明度

    # 标题 (可选，根据需要决定是否保留)
    # plt.title('ROC Curve', fontsize=14)

    plt.tight_layout()
    plt.savefig('roc_curve_styled.png', dpi=600, bbox_inches='tight')
    plt.show()

    # 预测概率输出
    class_probabilities = clf.predict_proba(X_test)

    result_df = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': y_pred,
        'Class_Probabilities': [list(probs) for probs in class_probabilities]
    })

    result_df.to_csv('probability_Fertility_rf.csv', index=False)
    print("Random Forest modeling completed!")

    def plot_custom_dependence(feature_name, shap_values_target, X_train_scaled, feature_names, scaler,
                               target_class_name):
        """
        绘制带有趋势线和自定义颜色的 SHAP 依赖图
        """
        # 1. 找到特征的索引
        try:
            feature_idx = feature_names.index(feature_name)
        except ValueError:
            print(f"Error: Feature '{feature_name}' not found in feature_names.")
            return

        # 2. 数据还原 (反归一化)
        # 我们需要原始的物理数值作为 X 轴，而不是标准化后的数值
        # scaler.inverse_transform 需要输入完整的矩阵，所以我们还原整个 X_train
        X_train_original = scaler.inverse_transform(X_train_scaled)

        # 提取当前特征的 X (特征值) 和 Y (SHAP值)
        x_data = X_train_original[:, feature_idx]
        y_data = shap_values_target[:, feature_idx]

        # 3. 创建画布
        fig, ax = plt.subplots(figsize=(8, 5))

        # 4. 绘制散点图
        # c=y_data 表示根据 SHAP 值的大小来着色 (与参考图一致)
        # cmap='RdYlBu_r' 使用红-黄-蓝反转色谱 (红色为高，蓝色为低)
        sc = ax.scatter(x_data, y_data, c=y_data, cmap='RdYlBu_r',
                        s=20, alpha=0.8, edgecolor='none')

        # 5. 添加趋势线 (LOWESS 平滑)
        # frac 参数控制平滑度，数值越小越敏感，越大越平滑 (0.1 - 0.3 通常比较好)
        z = lowess(y_data, x_data, frac=0.2)
        # lowess 返回的是按 x 排序的 (x, y) 点
        ax.plot(z[:, 0], z[:, 1], color='black', linestyle='--', linewidth=2, label='Trend')

        # 6. 添加水平零线
        ax.axhline(0, color='gray', linestyle='-.', linewidth=1, alpha=0.6)

        # 7. 设置标签和标题
        ax.set_xlabel(f"{feature_name}", fontsize=16)
        ax.set_ylabel("SHAP value", fontsize=16)
        # ax.set_title(f"Dependence Plot: {feature_name} (Class: {target_class_name})", fontsize=14)
        ax.legend(fontsize=14)  # 图例字体大小

        # 8. 设置坐标轴刻度标签字体大小
        # 方法1：直接设置刻度标签字体大小
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)

        # 8. 添加颜色条
        cbar = plt.colorbar(sc, ax=ax)
        # cbar.set_label('SHAP value', rotation=270, labelpad=15)

        # 保存图片
        safe_name = feature_name.replace('/', '_').replace(' ', '_').replace('*', '#')
        plt.tight_layout()
        plt.savefig(f'dependence_plot/Class_4/dependence_plot_{safe_name}.png', dpi=600)
        # plt.show()


    # ==========================================
    # 执行绘图逻辑
    # ==========================================

    # 0. 准备数据
    # 假设你关注的是类别 0 (根据你之前的 Target Class: I)
    target_class_index = 3
    target_class_label = label_encoder.inverse_transform([target_class_index])[0]

    # 处理 shap_values 格式 (如果是列表取列表，如果是数组取切片)
    if isinstance(shap_values, list):
        shap_vals_target = shap_values[target_class_index]
    else:
        # 假设它是 numpy array [samples, features, classes]
        shap_vals_target = shap_values[:, :, target_class_index]

    # 1. 选择你想画的特征
    # 通常我们画最重要的前几个特征，比如根据你的蜂群图，最重要的可能是 'Hf' 或 'Ti'
    top_features = feature_names  # 你可以修改这个列表

    for feat in top_features:
        plot_custom_dependence(
            feature_name=feat,
            shap_values_target=shap_vals_target,
            X_train_scaled=X_train,  # 传入标准化后的 X_train
            feature_names=feature_names,
            scaler=scaler,  # 传入你的 StandardScaler 对象
            target_class_name=target_class_label
        )
    def plot_all_dependencies_in_one_grid(feature_names, shap_values_target, X_train_scaled, scaler,
                                          target_class_name, cols=5, figsize=(25, 20)):
        """
        将所有特征的SHAP依赖图绘制在一张大图上
        """
        # 1. 计算需要的行数
        n_features = len(feature_names)
        rows = int(np.ceil(n_features / cols))

        # 2. 数据还原 (反归一化)
        X_train_original = scaler.inverse_transform(X_train_scaled)

        # 3. 创建画布和子图
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()  # 将二维axes数组展平为一维，便于循环

        # 4. 为每个特征绘制依赖图
        for idx, feature_name in enumerate(feature_names):
            ax = axes[idx]

            # 找到特征的索引
            try:
                feature_idx = feature_names.index(feature_name)
            except ValueError:
                print(f"Error: Feature '{feature_name}' not found in feature_names.")
                continue

            # 提取当前特征的 X (特征值) 和 Y (SHAP值)
            x_data = X_train_original[:, feature_idx]
            y_data = shap_values_target[:, feature_idx]

            # 绘制散点图
            sc = ax.scatter(x_data, y_data, c=y_data, cmap='RdYlBu_r',
                            s=15, alpha=0.7, edgecolor='none')

            # 添加趋势线 (LOWESS 平滑)
            if len(x_data) > 1:  # 确保有足够的数据点
                z = lowess(y_data, x_data, frac=0.2)
                ax.plot(z[:, 0], z[:, 1], color='black', linestyle='--',
                        linewidth=1.5, label='Trend')

            # 添加水平零线
            ax.axhline(0, color='gray', linestyle='-.', linewidth=1, alpha=0.6)

            # 设置标题和标签
            ax.set_xlabel(f"{feature_name}", fontsize=12)
            # ax.set_xlabel("Feature Value", fontsize=10)
            ax.set_ylabel("SHAP Value", fontsize=10)

            # 设置刻度标签字体大小
            ax.tick_params(axis='both', which='major', labelsize=9)

            # 为每个子图添加颜色条
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        # 5. 隐藏多余的子图（如果特征数不能被cols整除）
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        # 6. 添加主标题
        # fig.suptitle(f'SHAP Dependence Plots for All Features (Class: {target_class_name})',
        #              fontsize=16, y=0.98)

        # 7. 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间

        # 8. 保存图片
        plt.savefig(f'dependence_plot/Class_4/all_dependence_plots_grid.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


    # ==========================================
    # 执行绘图逻辑
    # ==========================================

    # 0. 准备数据
    target_class_index = 3
    target_class_label = label_encoder.inverse_transform([target_class_index])[0]

    # 处理 shap_values 格式
    if isinstance(shap_values, list):
        shap_vals_target = shap_values[target_class_index]
    else:
        shap_vals_target = shap_values[:, :, target_class_index]

    # 调用函数绘制所有依赖图在一张大图上
    plot_all_dependencies_in_one_grid(
        feature_names=feature_names,
        shap_values_target=shap_vals_target,
        X_train_scaled=X_train,
        scaler=scaler,
        target_class_name=target_class_label,
        cols=5,  # 每行5个子图
        figsize=(25, 20)  # 调整画布大小以适应更多子图
    )