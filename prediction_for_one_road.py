#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from bokeh.plotting import figure,  show, output_notebook, output_file
from bokeh.layouts import gridplot
from scipy import stats as st
from scipy import linalg
import timeit

from sklearn.mixture import GaussianMixture # нужна для бимодальности

# output to static HTML file
output_notebook()


# In[ ]:


# определяем функцию, которая будет нам рисовать гистограммы.  
def make_hist_plot(inp, title = '', plot_width = 240, plot_height = 240, bins = 50):
    hist, edges = np.histogram(inp, density=True, bins=bins)
    p = figure(title=title, background_fill_color="#fafafa", plot_width = plot_width, plot_height = plot_height)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5)
    return p



# определяем функцию, которая будет рисовать scatterplot (диаграмма рассеивания)
def make_scat_plot(inp1,inp2, x_range1, x_range2, y_range1, y_range2, color='navy', title ='', x_title = '', y_title = ''):
    p = figure(x_range=(x_range1,x_range2), y_range=(y_range1,y_range2))
    p.title.text = title
    p.xaxis.axis_label = x_title
    p.yaxis.axis_label = y_title

    p.circle(inp1,inp2, size=1, color = color, alpha=0.5)
    return p


# In[ ]:


class Distributions:
    def __init__(self, p_value=0, distrib_names=" ", params=0, mean=0, sigma = 0, models = 0, groups = 0):
        self.p_value = p_value
        self.distrib_names = distrib_names
        self.mean = mean
        self.sigma = sigma
        self.models = models
        self.groups = groups


# In[ ]:


# dataFrame - данные по скоростям (без всяких дней/часов/лет)
# length - длины дорог
# N число симуляций
# lower_bound - округление скорости. Это нужно для нормальных, чтобы небыло отрицательных чисел и около нуля. 
def make_prediction(DataFrame,length,N = 5000 ,lower_bound = 1, length_bound = 0.1, unimodal_bound = 0.05,
                    bimodal_bound = 0.05, make_bi = True, make_tri = True, coef_multimodal = 1, number_group = 1, Distribution_names = ["LN","LN2","LN3","N","N2","N3"]):

    starttime = timeit.default_timer()
    if np.min(np.sum(DataFrame.notna()).values) < 2:
        return print('EST STOLBEC U KOTOROGO MENSHE CHEM 2 ELEMENTA! NICHEGO POSCHITAT DLYA NEGO NELZA' )


    df = DataFrame.copy()
    numpy_data = df.to_numpy()

    numpy_log_data = np.log(numpy_data)

    df_log = np.log(df) #дата сет из логарифмов
    shape = df.shape

    #Оценка параметров для логнормального и нормального распределения
    sigma = np.std(df, ddof = 1).values # стандартные отклонения
    mean = np.mean(df).values  # математические ожидания

    sigma_log = np.std(df_log, ddof = 1).values # стандартные отклонения для логарифмов
    mean_log = np.mean(df_log).values  # математические ожидания для логарифмов

    

    #так мы будем хранить названия распределений
    distribution_names = dict(  zip([0,1,2,3,4,5],  Distribution_names) )
    models = dict()
    # тест Колмогорова на нормальность и логнормальность
    test_lognorm = [st.kstest(df.iloc[:,i].dropna(), 'lognorm', args = (sigma_log[i], 0 , np.exp(mean_log[i] ) ) ).pvalue
                    for i in range(shape[1])]
    test_norm = [st.kstest(df.iloc[:,i].dropna(), 'norm', args = (mean[i],sigma[i])).pvalue
                 for i in range(shape[1])]

    pvalue_uni = np.maximum(test_norm, test_lognorm) # берём максимум из одномодальных пи значений

    test_length_bound = length < length_bound # те участки, у которых длина меньше length_bound

    # булевкая маска сравнения одномодальных пи значений с граничным значением
    masked_distribs = pvalue_uni < unimodal_bound
    masked_distribs = masked_distribs.reshape(-1,1)

    # вычисляем бимодальность при нормальных компонентах
    test_norm_bi, models_norm_bi = multimodalPValue(numpy_data,2,test_length_bound, masked_distribs,make_bi, coef_multimodal)
    # вычисляем бимодальность при логнормальных компонентах
    test_lognorm_bi, models_lognorm_bi = multimodalPValue(numpy_log_data,2, test_length_bound, masked_distribs,make_bi,coef_multimodal)

    pvalue_bi = np.maximum(test_norm_bi, test_lognorm_bi) # берём максимум из бимодаыльных пи значений

    # булевкая маска сравнения бимодальных пи значений с граничным значением
    masked_bi_distribs = pvalue_bi < bimodal_bound
    masked_distribs = np.append(masked_distribs, (masked_bi_distribs.reshape(-1,1)),axis=1)

    # вычисляем тримодальность при нормальных компонентах
    test_norm_tri, models_norm_tri = multimodalPValue(numpy_data,3,test_length_bound, masked_distribs, make_tri, coef_multimodal)
    # вычисляем тримодальность при логнормальных компонентах
    test_lognorm_tri, models_lognorm_tri = multimodalPValue(numpy_log_data,3, test_length_bound, masked_distribs, make_tri, coef_multimodal)

    pvalue_tri = np.maximum(test_norm_tri, test_lognorm_tri) # берём максимум из тримодальных пи значений

    pvalue_multi = np.zeros(pvalue_tri.shape)
    mask_tri = pvalue_tri > 0

    test_uni = np.array(np.array(test_lognorm) > np.array(test_norm)) # True, если pvalue у логнормального больше, иначе False
    test_bi = np.array(np.array(test_lognorm_bi) > np.array(test_norm_bi)) # тоже самое для бимодального
    test_tri = np.array(np.array(test_lognorm_tri) > np.array(test_norm_tri)) # тоже самое для тримодального

    test_bimodal_bound = np.max( [test_norm_bi,test_lognorm_bi], axis =0) > bimodal_bound # те участки у которых пи значение для бимодального меньше bimodal_bound

    # создаём массивы из объектов типа GaussianMixture
    vectorized = np.vectorize(GaussianMixture)
    models_bi = np.empty(shape[1], dtype=object)
    models_bi[:,] = vectorized(models_bi)

    models_tri = np.empty(shape[1],dtype=object)
    models_tri[:,] = vectorized(models_tri)

    models_multi =  np.empty(shape[1],dtype=object)
    models_multi[:,] = vectorized(models_tri)

    # берём наилучишую модель при бимодальном распределении
    models_bi[test_bi] = models_lognorm_bi[test_bi]
    models_bi[~test_bi] = models_norm_bi[~test_bi]

    # берём наилучишую модель при тримодальном распределении
    models_tri[test_tri] = models_lognorm_tri[test_tri]
    models_tri[~test_tri] = models_norm_tri[~test_tri]

    #создаем итоговый селекшион
    test_result = np.zeros(shape[1])
    
    #одномодальность
    test_result = np.vectorize( {True : 0, False : 3 }.get )(test_uni)

    #бимодальность
    if np.max(test_bimodal_bound) == True:
        test_result[test_bimodal_bound]  = np.vectorize( {True : 1, False : 4 }.get )             (test_bi[test_bimodal_bound])
        models_multi[test_bimodal_bound] = models_bi[(test_bimodal_bound)]
        pvalue_multi[test_bimodal_bound] = pvalue_bi[(test_bimodal_bound)]
        
    #тримодальность
    if np.max(~test_bimodal_bound & mask_tri) == True:
        test_result[~test_bimodal_bound & mask_tri] = np.vectorize( {True : 2, False : 5 }.get )             (test_tri[~test_bimodal_bound & mask_tri])

        # берём наилучшие модели среди всех мультимодальных распределений
        models_multi[~test_bimodal_bound & mask_tri] = models_tri[~test_bimodal_bound & mask_tri]
        pvalue_multi[~test_bimodal_bound & mask_tri] = pvalue_tri[~test_bimodal_bound & mask_tri]



    #записываем полученные результаты в класс Distributions
    distributions = Distributions()

    distributions.distrib_names = np.vectorize(distribution_names.get)(test_result)

    # инициализация параметров
    distributions.p_value = np.zeros(distributions.distrib_names.shape)
    distributions.mean = np.zeros(distributions.distrib_names.shape)
    distributions.sigma = np.zeros(distributions.distrib_names.shape)

    test_norm = np.asanyarray(test_norm, dtype=np.float32)
    test_lognorm = np.asanyarray(test_lognorm, dtype=np.float32)

    # записывем одномодальное нормальное
    mask_by_name = distributions.distrib_names == "N"
    distributions.p_value[mask_by_name] = test_norm[mask_by_name]
    distributions.mean[mask_by_name] = mean[mask_by_name]
    distributions.sigma[mask_by_name] = sigma[mask_by_name]

    # записывем одномодальное логнормальное
    mask_by_name = distributions.distrib_names == "LN"
    distributions.p_value[mask_by_name] = test_lognorm[mask_by_name]
    distributions.mean[mask_by_name] = mean_log[mask_by_name]
    distributions.sigma[mask_by_name] = sigma_log[mask_by_name]

    # записывем мультимодальные распределения
    mask_by_name = np.logical_and(distributions.distrib_names != "N", distributions.distrib_names != "LN")
    distributions.p_value[mask_by_name] = pvalue_multi[mask_by_name]

    vectorized = np.vectorize(GaussianMixture)
    distributions.models = np.empty(shape[1], dtype=object)
    distributions.models[:,] = vectorized(distributions.models)
    distributions.models[mask_by_name] = models_multi[mask_by_name]
    
    distributions = uniteMultimodals(distributions)

    test_result = test_result == 0

    ### test_result  [0,1,2,3,4,5]

    #сохраняем логарифмы у тех линков, у которых логнормальное распределение показало большее пи-значение.
    #Таким образом, мы предполагаем, что итоговый вектор имеет многомерное нормальное распределение.

    df.loc[:,test_result] = df_log.loc[:,test_result]

    #находим Матрицу ковариации и вектор средних
    #cov  = EmpiricalCovariance().fit(df).covariance_ #так считается в sklearn. Должно быть лучше, но надо бороться с NaN
    cov = df.cov().values     #так считается матрица ковариации в pandas. Сравнение с EmpiricalCovariance не проводил.
    # есть функция statsmodels.stats.correlation_tools.cov_nearest, которая находит ближайжую положительно определенную матрицу. Нужно проверить как она работает.
    if np.isnan(cov).any() == True:
        print('THERE ARE NOT ENOUGH DATA TO ESTIMATE COVARIANCE.')
        cov[ np.isnan(cov)] = 0


    mean[test_result] = mean_log[test_result]

    # вот тут должен появится тест на независимость. Либо тест на многомерную нормальность.
    #....


    # симулируем итоговое распределение
    multi_norm = st.multivariate_normal.rvs(mean = mean, cov = cov , size = N) # симулируем многомерное нормальное распределение

    multi_norm[:,test_result] = np.exp(multi_norm[:,test_result]) #применяем экспоненту к тем участкам, где должна быть логнормальность

    #multi_norm[(~test_result) & (multi_norm < 1)  ] = 1  # такой строкой можно заменить у нормальных распределений все значения <1  на 1.


    #время в секундах (тупо беру скорости, которые больше 1. )
    prediction = 3600*np.sum(  length /   multi_norm[np.min(multi_norm, axis = 1) > lower_bound], axis = 1)

    print('Время работы функции make_prediction', np.round(timeit.default_timer() - starttime), 'секунд') # считаем время работы функции
        
    # выводим выборку, логнормальное/нормальное распределение, результаты тестов
    
    return prediction


# In[ ]:


# функция распределения смеси нормальных распределений
def mixNormCdf(x, model):
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()

    if len(x.shape) > 1:
        shape = x.shape[1]
    else:
        shape = 1

    x = x.reshape((shape,x.shape[0]))
    weights = model.weights_
    means = model.means_
    covars = model.covariances_
    mcdf = np.zeros(np.shape(x))
    for i in range(len(weights)):
        mcdf += weights[i] * st.norm.cdf(x, loc=means[i], scale=np.sqrt(covars[i]))

    return mcdf

# функция нахождения параметров для мультимодальных распределений
#parameters:
    # df - датафрейм, содержащий только скорости на различных участках
    # number_of_components - количество одномодальных распределений
    # test_length_bound - булевская маска сравнения длин маршрутов с пароговым значением   
    # masked_distribs - массив булевских масок сравнения пи значения разномодальных распределений с пороговыми значениями

#returns:
    # test_bimodal - посчитанные пи значения для мультимодальных распределений
    # models - модели с параметрами посчитанных распределений
    

def addWeekDay(df):
    week_day_column = np.zeros((df.shape[0],1))
    for i in range(df.shape[0]):
        week_day_column[i][0] = datetime.datetime(int(df["year"][i]),int(df["month"][i]),int(df["day"][i])).weekday()

    return week_day_column[:,0]




