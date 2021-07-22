import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender, bm25_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting=True):
        
        
        _popularity = data.groupby('item_id')['quantity'].sum().reset_index()
        _popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        self.top_5000 = _popularity.sort_values('n_sold', ascending=False).head(5000).item_id.tolist()
        
        self.user_item_matrix = self.prepare_matrix(data)
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        self.sparse_user_item = csr_matrix(self.user_item_matrix)
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    def prepare_matrix(self, data: pd.DataFrame):
        
        
        data.loc[~data['item_id'].isin(self.top_5000), 'item_id'] = 999999

        user_item_matrix = pd.pivot_table(data, 
                                          index='user_id', columns='item_id', 
                                          values='quantity',
                                          aggfunc='count', 
                                          fill_value=0
                                         )

        user_item_matrix = user_item_matrix.astype(float)
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    def fit(self, user_item_matrix, n_factors=256, regularization=0.04, iterations=5, num_threads=0):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                        regularization=regularization,
                                        iterations=iterations,  
                                        num_threads=num_threads)
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr())
        
        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        res = [self.id_to_itemid[rec[0]] for rec in 
                    self.model.recommend(userid=self.userid_to_id[user], 
                                         user_items=self.sparse_user_item,   
                                         N=N, 
                                         filter_already_liked_items=False, 
                                         filter_items=None, 
                                         recalculate_user=True)]
        
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_similar_users_recommendation(self, user, N=5):
        """
        Рекомендуем топ-N товаров, среди купленных похожими юзерами
        """
        user_tup = self.model.similar_users(self.userid_to_id[user], N=N+1)[1:]
        users = [usr[0] for usr in user_tup]
        
        tmp_res = []
        for usr in users:
            tmp_res.extend(self.model.recommend(userid=self.userid_to_id[usr], 
                        user_items=self.sparse_user_item,   
                        N=1,  # берем по одной вещи от каждого похожего юзера
                        filter_already_liked_items=False, 
                        filter_items=None, 
                        recalculate_user=False))
        res = [itm[0] for itm in tmp_res]

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res