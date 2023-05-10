from sklearn import metrics

class Performance(object):

    def get_performance(self, dataset, instance_model):
        silhouette_value = self.__get_silhouette(dataset, instance_model)
        calinski_value = self.__get_calinski_harabasz(dataset, instance_model) 
        davies_value = self.__get_davies_bouldin(dataset, instance_model)

        dict_response = {"silhouette_value": silhouette_value,
                        "calinski_value": calinski_value,
                        "davies_value" : davies_value
                        }
        if dict_response["silhouette_value"]:
            return dict_response
        
        
    def __get_silhouette(self, dataset, instance_model):

        try:
            response = metrics.silhouette_score(dataset, instance_model.labels_, metric='euclidean')
            return response
        except:
            return False
    
    def __get_calinski_harabasz(self, dataset, instance_model):
        try:
            response = metrics.calinski_harabasz_score(dataset, instance_model.labels_)
            return response
        except:
            return False
    
    def __get_davies_bouldin(self, dataset, instance_model):
        try:
            response = metrics.davies_bouldin_score(dataset, instance_model.labels_)
            return response
        except:
            return False