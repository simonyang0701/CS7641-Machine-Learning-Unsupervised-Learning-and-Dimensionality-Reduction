import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

def em(X_train, X_test, y_train, y_test, init_means, no_iter = 1000, component_list = [3,4,5,6,7,8,9,10,11], num_class = 7, toshow = 1):


	array_aic = []
	array_bic = []
	array_homo =[]
	array_comp = []
	array_sil = []
	array_avg_log = []


	for num_classes in component_list:

		clf = GaussianMixture(n_components=num_classes,covariance_type='spherical', max_iter=no_iter, init_params= 'kmeans')
		#     clf = KMeans(n_clusters= num_classes, init='k-means++')

		clf.fit(X_train)

		y_test_pred = clf.predict(X_test)
		#Per sample average log likelihood
		avg_log = clf.score(X_test)
		array_avg_log.append(avg_log)


		#AIC on the test data
		aic = clf.aic(X_test)
		array_aic.append(aic)

		#BIC on the test data
		bic = clf.bic(X_test)
		array_bic.append(bic)

		#Homogenity score on the test data
		homo = metrics.homogeneity_score(y_test, y_test_pred)
		array_homo.append(homo)

		#Completeness score
		comp = metrics.completeness_score(y_test, y_test_pred)
		array_comp.append(comp)

		#Silhoutette score
		sil = metrics.silhouette_score(X_test, y_test_pred, metric='euclidean')
		array_sil.append(sil)



	#Generating plots

	fig1,ax1 = plt.subplots()
	ax1.plot(component_list, array_aic)
	ax1.plot(component_list, array_bic)
	plt.legend(['AIC', 'BIC'])
	plt.xlabel('Number of clusters')
	plt.title('AIC/BIC curve for Expected Maximization')

	fig2,ax2 = plt.subplots()
	ax2.plot(component_list, array_homo)
	ax2.plot(component_list, array_comp)
	ax2.plot(component_list, array_sil)
	plt.legend(['homogenity','completeness','silhoutette'])
	plt.xlabel('Number of clusters')
	plt.title('Performance evaluation scores for Expected Maximization')


	fig3, ax3 = plt.subplots()
	ax3.plot(component_list, array_avg_log)
	plt.xlabel('Number of clusters')
	plt.title('Per sample average log likelihood for Expected Maximization')


	if(toshow == 1):
		plt.show()

	#Training and testing accuracy for K = number of classes
	
	clf = GaussianMixture(n_components=num_class ,covariance_type='spherical', max_iter=no_iter, init_params= 'kmeans')

	#Assigning the initial means as the mean feature vector for the class
	clf.means_init = init_means

	clf.fit(X_train)

	#Training accuracy
	y_train_pred = clf.predict(X_train)
	train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
	print('Training accuracy for Expected Maximization for K = {}:  {}'.format(num_class, train_accuracy))

	#Testing accuracy
	y_test_pred = clf.predict(X_test)
	test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
	print('Testing accuracy for Expected Maximization for K = {}:  {}'.format(num_class, test_accuracy))

	return component_list, array_aic, array_bic, array_homo, array_comp, array_sil, array_avg_log

	
def kmeans(X_train, X_test, y_train, y_test, init_means, no_iter = 1000, component_list =[3,4,5,6,7,8,9,10,11], num_class = 7, toshow=  1):

	array_homo =[]
	array_comp = []
	array_sil = []
	array_var = []

	for num_classes in component_list:
	    
	    clf = KMeans(n_clusters= num_classes, init='k-means++')
	    
	    clf.fit(X_train)
	    
	    y_test_pred = clf.predict(X_test)
	    
	      
	    #Homogenity score on the test data
	    homo = metrics.homogeneity_score(y_test, y_test_pred)
	    array_homo.append(homo)
	    
	    #Completeness score
	    comp = metrics.completeness_score(y_test, y_test_pred)
	    array_comp.append(comp)
	    
	    #Silhoutette score
	    sil = metrics.silhouette_score(X_test, y_test_pred, metric='euclidean')
	    array_sil.append(sil)

	    #Variance explained by the cluster
	    var = clf.score(X_test)
	    array_var.append(var)
	    
	    

	#Generating plots
	fig4,ax4 = plt.subplots()
	ax4.plot(component_list, array_homo)
	ax4.plot(component_list, array_comp)
	ax4.plot(component_list, array_sil)
	plt.legend(['homogenity','completeness','silhoutette'])
	plt.xlabel('Number of clusters')
	plt.title('Performance evaluation scores for KMeans')


	fig5, ax5 = plt.subplots()
	ax5.plot(component_list, array_var)
	plt.title('Variance explained by each cluster for KMeans')
	plt.xlabel('Number of cluster')

	if(toshow == 1):

		plt.show()


	#Training and testing accuracy for K = num_class

	#Assigning the initial means as the mean feature vector for the class
	init_mean = init_means
	clf = KMeans(n_clusters= num_class, init = init_mean)

	clf.fit(X_train)

	#Training accuracy
	y_train_pred = clf.predict(X_train)
	train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
	print('Training accuracy for KMeans for K = {}:  {}'.format(num_class, train_accuracy))

	#Testing accuracy
	y_test_pred = clf.predict(X_test)
	test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
	print('Testing accuracy for KMeans for K = {}:  {}'.format(num_class, test_accuracy))


	return component_list, array_homo, array_comp, array_sil, array_var