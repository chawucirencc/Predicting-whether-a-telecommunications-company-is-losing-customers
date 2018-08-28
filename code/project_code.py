#!/usr/bin/env.python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import itertools
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

"""
对电信公司3333条客户信息做出分析和预测，其中数据有20个特征，一个类别项，
通过对数据的总体描述，特征的可视化，以及某些特征与类别项的关联分析，发现特征的特点和特征与最终结果的关系，
然后对原数据的object进行转换，并对最终的分类结果进行分离，由于这3000余条信息不存在空值所以不需要对数据进行
额外处理，然后对六种算法以及四种集成学习分方法进行比较，在使用默认参数的条件下选出相对较好的算法建立模型，
最后使用梯度提升算法建立模型分类，随后对所建立的模型进行评估，使用精准率，准确率，召回率F1值等，通过可视化的方法
对模型进行评估，最后对建立的模型保存到本地，并且重新加载。
"""
def load_data(file_path):
    """打开文件，并对每列重新命名， 返回的是原始数据"""
    names = ['state', 'acc_length', 'area', 'ph_num', 'inter_plan', 'vm_plan', 'num_vm_message', 'day_min',
             'day_calls', 'day_charge', 'eve_min', 'eve_calls', 'eve_charge', 'night_min', 'night_calls',
             'night_charge', 'inter_min', 'inter_calls', 'inter_charge', 'cus_ser_calls', 'churn']
    data = pd.read_csv(file_path, names=names, header=0)
    pd.set_option('display.max_columns', 30)
    pd.set_option('precision', 3)
    return data


def check_data_feature(all_data):
    """查看多特性可以调用不同的函数"""
    print(all_data.shape)                                           # 查看数据的行数和列数
    print(all_data.groupby('churn').size())                         # 查看数据属性分类
    # print('整体描述：'+'-'*80+'\n', all_data.describe())           # 查看数据的整体描述
    # print('数据的自相关性：'+'-'*80+'\n', all_data.corr())          # 查看数据的自相关性
    # print('数据的属性类型：'+'-'*80)       # 设置分割线
    # print(all_data.info())               # 查看数据的属性类型（其中有四项为object，一项为bool，最后要讲其转化为数值型）


def draw_feature_plot(all_data):
    """
    对数据特征进行可视化，由图可知，在电话时长，电话次数，
    电话费用以及国际电话等方面其频率图几乎都符合于高斯分布。
    """
    fig = plt.figure()
    fig.set(alpha=0.8)
    fig.add_subplot(121)
    all_data['churn'].value_counts().plot(kind='bar')   # 对churn进行统计，查看流失与未流失的用户数量
    plt.title('churn True or False')
    fig.add_subplot(122)
    all_data['cus_ser_calls'].value_counts().plot(kind='bar')   # 对拨打客服电话次数进行统计
    plt.title('customer service calls times')
    # fig2、，可以清楚的看清特征的分布，几乎都服从高斯分布
    fig2 = plt.figure()       # 这里是对用户在各个时间段的分钟数，拨打次数，费用以及国际用户的三项作的频率图
    fig2.set(alpha=0.8)
    fig2.tight_layout()
    fig2.subplots_adjust(wspace=0.7, hspace=0.7)
    name = ['day_min', 'day_calls', 'day_charge', 'eve_min', 'eve_calls', 'eve_charge',
            'night_min', 'night_calls', 'night_charge', 'inter_min', 'inter_calls', 'inter_charge']
    for i, n in enumerate(name, start=1):
        fig2.add_subplot(4, 3, i)
        all_data[n].plot(kind='kde')
        plt.xlabel(n)
        plt.title('density of ' + n)
    plt.suptitle('feature frequency', fontsize=15)
    plt.show()


def feature_associated(all_data):
    """
    关于国际计划用户和拨打客服电话次数，通过对图的分析可知，在流失的客户中为国际计划的用户占比较多几乎为1/3,
    并且在拨打客服次数和最终结果的关联图上可知，在拨打3次客服之后流失的客户比较严重。
    """

    # 关于是否为国际计划用户流失图
    inter_yes = all_data['churn'][all_data['inter_plan'] == 'yes'].value_counts()
    inter_no = all_data['churn'][all_data['inter_plan'] == 'no'].value_counts()
    df_inter = pd.DataFrame({'inter plan': inter_yes, 'no inter plan': inter_no})
    df_inter.plot(kind='bar', stacked=True)
    plt.title('inter or no inter of churn', fontsize=15)
    plt.xlabel('inter or not inter')
    plt.ylabel('number')

    # 关于拨打客服次数和是否为流失客户的关系图
    cus_f = all_data['cus_ser_calls'][all_data['churn']==False].value_counts()
    cus_t = all_data['cus_ser_calls'][all_data['churn']==True].value_counts()
    df_cus = pd.DataFrame({'retain': cus_f, 'churn': cus_t})
    df_cus.plot(kind='bar', stacked=True)
    plt.title('customer service calls about churn', fontsize=15)
    plt.xlabel('customer service calls')
    plt.ylabel('numbers')
    plt.show()


def deal_data(all_data):
    """去掉无用的特征，并且将churn分离出数据，对object对象进行数值，对整个数据集进行标准化"""
    data_result = all_data['churn']
    y = np.where(data_result==True, 1, 0)
    new_inter = pd.get_dummies(all_data['inter_plan'], prefix='_inter_plan')
    new_vm_plan = pd.get_dummies(all_data['vm_plan'], prefix='_vm_plan')
    data_temp = pd.concat([all_data, new_inter, new_vm_plan], axis=1)
    to_drop = ['state', 'area', 'ph_num', 'inter_plan', 'vm_plan', 'churn']
    data_df = data_temp.drop(to_drop, axis=1)
    array = data_df.values
    std = StandardScaler()
    X = std.fit_transform(array)
    return X, y          # 返回特征数据和分类数据


def choose_algorithm(X, y):
    """
    比较六种算法的准确度，画出箱型图，在运行中系统对logistics和LDA都进行了变量之间共线的提示，
    而且从最终的箱型图来看，在这个问题上CART和SVM的两种算法都好于其他的几种算法，所以优先选择CART和SVM。"""
    models = list()
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    result = []
    names = []
    kfo = KFold(n_splits=10, shuffle=True, random_state=1)
    for name, model in models:
        cro_result = cross_val_score(model, X, y, scoring='accuracy', cv=kfo)
        result.append(cro_result)
        names.append(name)
        msg = ("{}:{:3f}\t({:3f})".format(name, cro_result.mean(), cro_result.std()))
        print(msg)
    fig = plt.figure()
    fig.suptitle('Algorithm to compare')
    ax = fig.add_subplot(111)
    plt.boxplot(result)
    plt.ylabel('accuracy')
    ax.set_xticklabels(names)
    plt.show()


def improve_result(x, y):
    """
    通过对比四种方法来选择那种方法提升算法, 通过结果可知梯度提升的方法准确率较高，
    而且算法本身也是用决策树进行集成学习，所以选择梯度提升算法建立模型。
    """
    kfo = KFold(n_splits=10, shuffle=True, random_state=1)
    rf_model = RandomForestClassifier()
    rf_result = cross_val_score(rf_model, x, y, cv=kfo)
    print('RandomForest方法:', rf_result.mean())
    gb_model = GradientBoostingClassifier()
    gb_result = cross_val_score(gb_model, x, y, cv=kfo)
    print('GradientBoosting方法:', gb_result.mean())
    ada_model = AdaBoostClassifier()
    ada_result = cross_val_score(ada_model, x, y, cv=kfo)
    print('AdaBoost方法:', ada_result.mean())
    bag_model = BaggingClassifier()
    bag_result = cross_val_score(bag_model, x, y, cv=kfo)
    print('Bagging方法:', bag_result.mean())


def create_model(x, y):
    """建立模型"""
    kfo = KFold(n_splits=10, shuffle=True, random_state=1)
    X_tranin, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    gb_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1)
    gb_model.fit(X_tranin, y_train)
    cv_result = cross_val_score(gb_model, X_tranin, y_train, cv=kfo)
    gb_model.fit(X_tranin, y_train)
    # print('模型准确率\t', gb_model.score(X_test, y_test))
    # print('交叉验证结果\t', cv_result.mean())
    return gb_model     # 返回模型


def plot_confusion_matrix(conf_mat, labels):
    """画出混淆矩阵函数"""
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion_matrix', fontsize=15)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, format(conf_mat[i, j]), horizontalalignment="center")


def assessment_model(x, y, model, labels):
    """
    评估模型，图一是画出混淆矩阵，图二是以计算精准率和召回率的阈值作为X轴画出精准率曲线和召回率两条曲线，
    在输出结果上看，精准率和召回率也都还可以，F1值为82.7%，最终的AUC为0.9。
    """
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    y_pre = model.predict(X_test)
    decision_fun = model.decision_function(X_test)
    fig_1 = plt.figure()    # 图一、画出混淆矩阵
    conf_mat = confusion_matrix(y_test, y_pre)
    plot_confusion_matrix(conf_mat, labels)
    print('精准率--->',precision_score(y_test, y_pre))
    print('F1值--->', f1_score(y_test, y_pre))
    print('召回率--->', recall_score(y_test, y_pre))
    pre, recall, thresholds = precision_recall_curve(y_test, decision_fun)
    fig_2 = plt.figure()        # 图二、画出精准率曲线和召回率曲线
    plt.plot(thresholds, pre[:-1])
    plt.plot(thresholds, recall[:-1])
    plt.title('precision_recall_curve', fontsize=15)
    plt.legend(['pre', 'recall'], loc='center right')
    fig_3 = plt.figure()        # 图三、画出召回率关于精准率的曲线（x轴为精准率，y轴为召回率）
    plt.plot(pre, recall)
    plt.xlabel('precision score')
    plt.ylabel('recall score')
    plt.title('recall score about precision score', fontsize=15)

    fig_4 = plt.figure()        # 图四、画出ROC曲线，并计算出AUC。
    fpr, tpr, thr = roc_curve(y_test, decision_fun)
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve', fontsize=15)
    print('AUC--->', auc(fpr, tpr))   # 计算AUC
    plt.show()


def save_model(model):
    """将模型以文件的形式保存到本地"""
    model_path = 'final_model.sav'
    pickle.dump(model, open(model_path, 'wb'))
    return model_path


def load_model(model_path):
    """读取本地的模型文件"""
    model_load = pickle.load(open(model_path, 'rb'))
    return model_load

def use_model(model_load, X, y):
    result = model_load.score(X, y)
    print(result)       # result = 0.9813981398139814


def main():
    file_path = 'data\churn.csv'
    labels = ['0', '1']
    all_data = load_data(file_path)
    check_data_feature(all_data)  # 查看数据特征、属性等
    draw_feature_plot(all_data)   # 画出特征频率图
    feature_associated(all_data)  # 画出特征关联图
    X, y = deal_data(all_data)      # 去除掉无用特征，将object数值化
    choose_algorithm(X, y)        # 比较算法
    improve_result(X, y)          # 提升结果、选择算法
    gbm_model = create_model(X, y)       # 创建模型
    assessment_model(X, y, gbm_model, labels=labels)  # 评估模型
    model_path = save_model(gbm_model)                # 保存模型
    model_load = load_model(model_path)               # 加载模型
    use_model(model_load, X, y)                       # 使用模型


main()
