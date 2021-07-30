import pandas as pd
import time
import numpy as np
import math
import pika
import json
from bson import json_util
import sys

class MatrixFactorization:
    def __init__(self,l,c,d,runPerRound):
        self.l = l
        self.c = c
        self.d = d
        self.runPerRound = runPerRound
        self.userRanking = dict()
    
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
        for movieId in contentIds:
            contentRows[movieId] = i
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
    
    def normalizeByRows(self,M):
        u = np.zeros(M.shape[0])
        for i in range(M.shape[0]):
            m = M[i]
            elems = m[m != 0]
            if len(elems) == 0:
                u[i] = 0
            else:
                u[i] = np.mean(elems)
            mask = M[i] != 0
            M[i, mask] = np.subtract(M[i, mask], u[i])
        return u
    
    def normalizeByColumns(self,M):
        i = self.normalizeByRows(M.transpose())
        M = M.transpose()
        return i
    
    def restoreNormalization(self,P, rowAvg, columnAvg):
        P = np.add(P, columnAvg)
        P = P.transpose()
        P = np.add(P, rowAvg)
        P = P.transpose()
        return P
    
    def fill(self,M, c, a, d):
        div = a/d
        base = float(math.sqrt(abs(a/d)))
        base = math.copysign(base, div)
        if c == 0:
            M.fill(base)
        else:
            M = (np.random.rand(M.shape[0],M.shape[1]) - 0.5) * 2*c + base
        return M
    
    def getPermutation(self,M):
        nonzeros = np.transpose(np.nonzero(M))
        np.random.shuffle(nonzeros)
        return nonzeros
    
    def improve(self,M, U, V, i, j, l):
        p = np.copy(U[i])
        V = V.transpose()
        q = np.copy(V[j])
        err = M[i,j] - np.dot(p, q)
        U[i] = p + 2*l*err*q
        V[j] = q + 2*l*err*p
        V = V.transpose()
        return 2*l*err*q
    
    def trainMatrix(self,M0):
        print("trainMatrix")
        t0 = time.time()

        if len(M0[M0 != 0]) == 0:
            avg = 0
        else:
            avg = np.mean(M0[M0 != 0])

        M = np.copy(M0)
        rowAvg = self.normalizeByRows(M)
        colAvg = self.normalizeByColumns(M)
        U = self.fill(np.zeros((M.shape[0], self.d)), self.c, avg, self.d)
        V = self.fill(np.zeros((self.d, M.shape[1])), self.c, avg, self.d)
        perm = self.getPermutation(M0)

        for n in range(self.runPerRound):
            for p in perm:
                self.improve(M, U, V, p[0], p[1], self.l)

        P = np.dot(U,V)
        P = self.restoreNormalization(P, rowAvg, colAvg)

        t1 = time.time()
        print(t1-t0)

        return (U,V,P,rowAvg,colAvg)
    
    def initializeModel(self,ratings):
        t0 = time.time()

        self.uIds = self.userIds(ratings)
        self.cIds = self.contentIds(ratings)

        self.uRows = self.userRows(self.uIds)
        self.cRows = self.contentRows(self.cIds)
        
        self.reverseUserRows = self.reverseDict(self.uRows)
        self.reverseContentRows = self.reverseDict(self.cRows)

        self.M = self.createMatrix(ratings, self.uIds, self.cIds, self.uRows, self.cRows)
        (self.U,self.V,self.P,self.rowAvg,self.colAvg) = self.trainMatrix(self.M)

        t1 = time.time()
        print("Training time: " + str(t1-t0))
        
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

                f = open("log/updateMF.log", "a")
                f.write(json.dumps(d)+'\n')
                f.close()
            
    def updateModel(self,data):
        oldShape = self.M.shape
        coords = []
        for d in data:
            coords.append(self.addRating(d))
        self.reverseUserRows = self.reverseDict(self.uRows)
        self.reverseContentRows = self.reverseDict(self.cRows)
        np.random.shuffle(coords)
        self.retrain(data,oldShape,coords)
        
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
    
    def retrain(self,data,oldShape,coords):
        if len(self.M[self.M != 0]) == 0:
            avg = 0
        else:
            avg = np.mean(self.M[self.M != 0])
        M = np.copy(self.M)
        self.rowAvg = self.normalizeByRows(M)
        self.colAvg = self.normalizeByColumns(M)
        Uextra = self.fill(np.zeros((M.shape[0]-oldShape[0], self.d)), self.c, avg, self.d)
        Vextra = self.fill(np.zeros((self.d, M.shape[1]-oldShape[1])), self.c, avg, self.d)
        self.U = np.vstack((self.U, Uextra))
        self.V = np.hstack((self.V, Vextra))
        
        for n in range(self.runPerRound):
            for c in coords:
                self.improve(M, self.U, self.V, c[0], c[1], self.l)

        self.P = np.dot(self.U,self.V)
        self.P = self.restoreNormalization(self.P, self.rowAvg, self.colAvg)
        
    def getUserRank(self, uId):
        if uId in self.userRanking:
            return (True,self.userRanking[uId])
        else:
            return self.userRank(uId)
        
    def userRank(self, uId):
        userRanking = list()
        if uId not in self.uRows:
            return (False,self.getTopRanking())
        userId = self.uRows[uId]
        i = 0
        for v in self.P[userId]:
            userRanking.append((self.reverseContentRows[i], v))
            i += 1
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
        uRow = self.uRows[userId]
        cRow = self.cRows[contentId]
        pred = self.P[uRow,cRow]
        if pred > 5:
            pred = 5
        if pred < 1:
            pred = 1
        return (True, True, pred)
    
    def persist(self):
        print("saveMatrices")
        t0 = time.time()

        t1 = time.time()
        print(t1-t0)
        return {'M' : self.M, 'U': self.U, 'rowAvg': self.rowAvg, 'V': self.V, 'colAvg': self.colAvg, 'userRows': self.uRows, 'contentRows': self.cRows}
        
    def restore(self,storedData):
        print("restoreMatrices")
        t0 = time.time()
        
        self.M = storedData['M']
        self.U = storedData['U']
        self.V = storedData['V']
        self.rowAvg = storedData['rowAvg']
        self.colAvg = storedData['colAvg']
        self.uRows = storedData['userRows']
        self.cRows = storedData['contentRows']

        self.uIds = np.array(list(self.uRows.keys()))
        self.cIds = np.array(list(self.cRows.keys()))
        self.reverseUserRows = self.reverseDict(self.uRows)
        self.reverseContentRows = self.reverseDict(self.cRows)
        
        self.P = np.dot(self.U,self.V)
        self.P = self.restoreNormalization(self.P,self.rowAvg,self.colAvg)

        t1 = time.time()
        print(t1-t0)
