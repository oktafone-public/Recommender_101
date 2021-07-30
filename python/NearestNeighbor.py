import pandas as pd
import time
import numpy as np
import math
import pika
import json
from bson import json_util
import sys

class NearestNeighbor:
    def __init__(self,k):
        self.k = k
        self.userRanking = dict()
        
    def loadSampleContent(self):
        column_names = ["movieId", "title", "genre"]
        items = pd.read_table("movies.dat", encoding="latin1", sep="::", names=column_names, index_col="movieId", engine="python")
        items.genre = items.genre.str.replace("|", ", ")

        for i in items.index:
            items.genre[i] = items.genre[i].split(', ')

        return items
    
    def userIds(self,data):
        return data.ownerId.unique()
    
    def contentIds(self,data):
        return data.contentId.unique()
    
    def userRows(self,userIds):
        userRows = dict()

        i = 0
        for userId in userIds:
            userRows[userId] = i
            i += 1

        return userRows
    
    def contentRows(self,contentIds):
        contentRows = dict()

        i = 0
        for contentId in contentIds:
            contentRows[contentId] = i
            i += 1

        return contentRows
    
    def reverseDict(self,d):
        return {v:k for k,v in d.items()}
    
    def createMatrix(self,loaded, userIds, contentIds, userRows, contentRows):
        print("createMatrix")
        t0 = time.time()
        
        self.M = np.zeros((len(userIds), len(contentIds)))

        for r in zip(loaded['ownerId'], loaded['contentId'], loaded['rating']):
            self.M[userRows[r[0]], contentRows[r[1]]] = r[2]

        t1 = time.time()
        print(t1-t0)
        
        return self.M
    
    def standardDeviation(self,values):
        m = np.mean(values)
        s = np.sum(np.square(np.subtract(values,m)))
        return math.sqrt((1/(len(values)-1)*s))
    
    def covariance(self,x,y):
        xm = np.mean(x)
        ym = np.mean(y)
        xx = np.subtract(x,xm)
        yy = np.subtract(y,ym)
        s = np.sum(np.multiply(xx,yy))
        return (1/(len(x)-1))*s
    
    def pearson(self,x,y):
        sxy = self.covariance(x,y)
        sx = self.standardDeviation(x)
        sy = self.standardDeviation(y)
        return sxy/(sx*sy)
    
    def initializeGenres(self,items):
        genres = set()
        
        for i in items.index:
            genres |= set(items.genre[i])
                        
        return genres
    
    def setItemGenres(self):
        itemRows = self.contentRows(self.items.index)
        itemGenres = dict()
        
        for g in self.genres:
            self.items[g] = 0
            
        genreList = self.items['genre'].to_dict()
        for g in self.genres:
            gd = self.items[g].to_dict()
            for k in genreList:
                gd[k] = int(g in genreList[k])
            self.items[g] = gd.values()
            
        genreMatrix = self.items[list(self.genres)].as_matrix()
            
        for i in self.items.index:
            itemGenres[str(i)] = genreMatrix[itemRows[i]]
            
        return itemGenres
    
    def distanceMatrix(self):
        print("create distance matrix")
        t0 = time.time()
        
        l = len(self.itemGenres)
        D = np.zeros([l,l])
        coords = np.array(np.meshgrid(range(l),range(l))).T.reshape(-1,2)
        for i,j in coords:
            D[i,j] = self.pearson(self.itemGenres[self.reverseContentRows[i]],self.itemGenres[self.reverseContentRows[j]])
                
        t1 = time.time()
        print(t1-t0)
        return D
    
    def initializeModel(self,ratings):
        t0 = time.time()
        
        self.items = self.loadSampleContent()
        self.genres = self.initializeGenres(self.items)

        self.uIds = self.userIds(ratings)
        self.cIds = self.contentIds(ratings)

        self.uRows = self.userRows(self.uIds)
        self.cRows = self.contentRows(self.cIds)
        
        self.reverseUserRows = self.reverseDict(self.uRows)
        self.reverseContentRows = self.reverseDict(self.cRows)
        
        self.items = self.items[self.items.index.isin(list(map(int,self.cIds)))]
        self.itemGenres = self.setItemGenres()

        self.M = self.createMatrix(ratings, self.uIds, self.cIds, self.uRows, self.cRows)
        self.D = self.distanceMatrix()

        t1 = time.time()
        print("Training time: " + str(t1-t0))
        
    def kMostSimilar(self,contentId,rated):
        rank = list()
        cRow = self.cRows[contentId]
        Dt = self.D.transpose()
        contentDistances = Dt[cRow][0:cRow]
        contentDistances = np.append(contentDistances,self.D[cRow][cRow:self.D.shape[0]])
        for v in rated:
            if v != cRow:
                rank.append((str(self.reverseContentRows[v]), contentDistances[v]))
        rank.sort(key=lambda r: r[1], reverse=True)
        rank = rank[0:self.k]
        return rank
    
    def similarRated(self,userId, contentId):
        userRow = self.M[self.uRows[userId]]
        nonzero = np.nonzero(userRow)[0]
        if len(nonzero) == 0:
            return []
        movies = self.kMostSimilar(contentId,nonzero)
        return movies
    
    def kNN(self,userId, contentId):
        movies = self.similarRated(userId, contentId)
        if (len(movies) == 0):
            knn = 0
        else:
            userRow = self.M[self.uRows[userId]]
            num = sum([x[1]*userRow[self.cRows[x[0]]] for x in movies])
            den = sum([abs(x[1]) for x in movies])
            knn = num/den
        return knn
    
    def getUserRank(self, uId):
        if uId in self.userRanking:
            return (True,self.userRanking[uId])
        else:
            return self.userRank(uId)
        
    def userRank(self, uId):
        userRanking = list()
        if uId not in self.uRows:
            return (False,[])
        userId = self.uRows[uId]
        
        for cId in self.cIds:
            userRanking.append((cId, self.kNN(uId,cId)))
        
        userRanking.sort(key=lambda r: r[1], reverse=True)
        self.userRanking[uId] = userRanking
        return (True,userRanking)
    
    def userTop(self, uId,n):
        userRanking = self.getUserRank(uId)
        return (userRanking[0],userRanking[1][0:n])
    
    def userRankings(self,n):
        print("userRanking")

        userRanking = list()

        t0 = time.time()

        for userId in self.uIds:
            userRanking.append({'_id': userId, 'items': self.userTop(userId,n)[1]})

        t1 = time.time()
        print(t1-t0)

        return userRanking
    
    def getTopRanking(self):
        if 'top' in self.userRanking:
            return self.userRanking['top']
        else:
            return self.top()
    
    def top(self):
        colRank = list()
        i = 0
        for c in self.P.transpose():
            colRank.append(np.mean(c))
        topRanking = list()
        i = 0
        for c in colRank:
            topRanking.append((self.reverseContentRows[i], c))
            i += 1
        topRanking.sort(key=lambda r: r[1], reverse=True)
        self.userRanking['top'] = topRanking
        return topRanking
        
    def getRecommendation(self,n):
        ranking = self.userRankings(n)
        items = self.getTopRanking()
        items = items[0:n]
        ranking.append({'_id': 'top', 'items': items})
        return ranking
    
    def getPrediction(self,userId, contentId):
        if (userId not in self.uRows) or (contentId not in self.cRows):
            return (userId in self.uRows, contentId in self.cRows, 3)
        pred = self.kNN(userId, contentId)
        if pred > 5:
            pred = 5
        if pred < 1:
            pred = 1
        return (True, True, pred)
    
    def setUpdateProposer(self,updateProposer):
        if hasattr(self, 'updateProposer'):
            updateProposer.queue = self.updateProposer.queue
            updateProposer.unknownCount = self.updateProposer.unknownCount
            updateProposer.unknownItems = self.updateProposer.unknownItems
            updateProposer.unknownUsers = self.updateProposer.unknownUsers
        self.updateProposer = updateProposer
    
    def receiveData(self,data,recall,dcg,userKnown,itemKnown):
        if hasattr(self, 'updateProposer'):
            (shouldUpdate,updateData, reasons) = self.updateProposer.receiveData(data,recall,dcg,userKnown,itemKnown)
            if shouldUpdate:
                d = dict()
                d['Reasons'] = reasons
                d['New entries'] = len(updateData)
                t0 = time.time()

                self.updateModel(updateData)

                t1 = time.time()
                d['Update time'] = t1-t0

                f = open("log/updatekNN.log", "a")
                f.write(json.dumps(d)+'\n')
                f.close()
            
    def updateModel(self,data):        
        oldShape = self.M.shape
        oldContents = self.cIds
        coords = []
        for d in data:
            coords.append(self.addRating(d))
        self.reverseUserRows = self.reverseDict(self.uRows)
        self.reverseContentRows = self.reverseDict(self.cRows)
        self.retrainDistances(oldShape,oldContents,{x[1] for x in coords})
        
        self.userRanking = dict()
        
    def addRating(self,data):
        ownerId = data['ownerId']
        contentId = data['contentId']
        rating = data['rating']
        if not ownerId in self.uIds:
            self.uIds = np.append(self.uIds, ownerId)
            self.uRows[ownerId] = len(self.uRows)
            self.M = np.vstack([self.M, np.zeros((1, len(self.cIds)))])
        if not contentId in self.cIds:
            self.cIds = np.append(self.cIds, contentId)
            self.cRows[contentId] = len(self.cRows)
            self.M = np.hstack([self.M, np.zeros((len(self.uIds), 1))])
        self.M[self.uRows[ownerId], self.cRows[contentId]] = rating
        return (self.uRows[ownerId], self.cRows[contentId])    # Returns the coordinates of the new element
    
    def retrainDistances(self,oldShape,oldContents,contentsChanged):
        self.items = self.loadSampleContent()
        self.items = self.items[self.items.index.isin(list(map(int,self.cIds)))]
        newContents = [x for x in self.cIds if x not in oldContents]
        
        itemRows = self.contentRows(list(map(str,self.items.index)))
        self.itemGenres = self.setItemGenres()
        genreMatrix = self.items[list(self.genres)].as_matrix()
        
        for i in newContents:
            self.itemGenres[i] = genreMatrix[itemRows[i]]
        
        extraRows = np.zeros((len(self.cIds)-oldShape[1],oldShape[1]))
        extraCols = np.zeros((len(self.cIds),len(self.cIds)-oldShape[1]))
        self.D = np.vstack([self.D, extraRows])
        self.D = np.hstack([self.D, extraCols])
                
        for c in range(oldShape[1],len(self.cIds)):
            for i in range(len(self.cIds)):
                if i <= c:
                    self.D[i,c] = self.pearson(self.itemGenres[self.reverseContentRows[i]],self.itemGenres[self.reverseContentRows[c]])
                else:
                    self.D[c,i] = self.pearson(self.itemGenres[self.reverseContentRows[i]],self.itemGenres[self.reverseContentRows[c]])
    
    def persist(self):
        print("saveMatrices")
        t0 = time.time()

        t1 = time.time()
        print(t1-t0)
        return {'M' : self.M, 'D': self.D, 'userRows': self.uRows, 'contentRows': self.cRows, 'genres': self.genres}
        
    def restore(self,storedData):
        print("restoreMatrices")
        t0 = time.time()
        
        self.M = storedData['M']
        self.D = storedData['D']
        self.uRows = storedData['userRows']
        self.cRows = storedData['contentRows']
        self.genres = storedData['genres']
        

        self.uIds = np.array(list(self.uRows.keys()))
        self.cIds = np.array(list(self.cRows.keys()))
        self.reverseUserRows = self.reverseDict(self.uRows)
        self.reverseContentRows = self.reverseDict(self.cRows)

        t1 = time.time()
        print(t1-t0)
