import pandas as pd
import time
import numpy as np
import math
import pika
import json
from bson import json_util
import sys

class ItemToItem:
    def __init__(self,k):
        self.k = k
        self.itemRanking = dict()
        self.lastSeen = dict()
        self.items = dict()
    
    def contentIds(self,data):
        return data.contentId.unique()
    
    def contentRows(self,contentIds):
        contentRows = dict()

        i = 0
        for contentId in contentIds:
            contentRows[contentId] = i
            i += 1

        return contentRows
    
    def reverseDict(self,d):
        return {v:k for k,v in d.items()}
    
    def setLastSeen(self,loaded):
        for d in zip(loaded['ownerId'], loaded['contentId']):
            self.lastSeen[d[0]] = d[1]
            if d[1] not in self.items:
                self.items[d[1]] = set()
            self.items[d[1]].add(d[0])
            
    def createMatrix(self):
        l = len(self.items)
        M = np.zeros((l,l))

        coords = np.array(np.meshgrid(range(l),range(l))).T.reshape(-1,2)
        for i,j in coords:
            M[i,j] = self.jaccard(self.items[self.reverseContentRows[i]],self.items[self.reverseContentRows[j]])
            
        return M
        
    def initializeModel(self,ratings):
        t0 = time.time()
        
        self.cIds = self.contentIds(ratings)
        self.cRows = self.contentRows(self.cIds)
        self.reverseContentRows = self.reverseDict(self.cRows)
        
        self.setLastSeen(ratings)
        self.M = self.createMatrix()

        t1 = time.time()
        print("Training time: " + str(t1-t0))
        
    def jaccard(self,i,j):
        ij = {x for x in i if x in j}
        return len(ij)/(len(i)+len(j)-len(ij)+1)
        
    def getUserRecommendation(self,userId):
        if userId not in self.lastSeen:
            return (False, [])
        else:
            return (True, self.getItemRank(self.lastSeen[userId])
        
    def getRecommendation(self,n):
        ranking = self.itemRankings(n)
        return ranking
    
    def itemRankings(self,n):
        itemRanking = list()
        
        for itemId in self.cIds:
            itemRanking.append({'_id': itemId, 'items': self.itemTop(itemId,n)[1]})
            
        return itemRanking
    
    def itemTop(self,itemId,n):
        itemRanking = self.getItemRank(itemId)
        return (itemRanking[0],itemRanking[1][0:n])
    
    def getItemRank(self,itemId):
        if itemId in self.itemRanking:
            return (True,self.itemRanking[itemId])
        else:
            return self.itemRank(itemId)
        
    def itemRank(self,itemId):
        itemRanking = list()
        if itemId not in self.cRows:
            return (False,[])
        item = self.cRows[itemId]
        
        for cId in self.cIds:
            if cId != itemId:
                itemRanking.append((cId, self.M(self.cRows[itemId],self.cRows[cId])))
        
        itemRanking.sort(key=lambda r: r[1], reverse=True)
        self.itemRanking[itemId] = itemRanking
        return (True,itemRanking)
    
    def getPrediction(self,userId, contentId):
        return (True, contentId in self.cRows, 3)
    
    def setUpdateProposer(self,updateProposer):
        if hasattr(self, 'updateProposer'):
            updateProposer.queue = self.updateProposer.queue
            updateProposer.unknownCount = self.updateProposer.unknownCount
            updateProposer.unknownItems = self.updateProposer.unknownItems
            updateProposer.unknownUsers = self.updateProposer.unknownUsers
        self.updateProposer = updateProposer
    
    def receiveData(self,data,recall,dcg,userKnown,itemKnown):
        self.lastSeen[data['ownerId']] = data['contentId']
        self.items[data['contentId']].add(data['ownerId'])
        (shouldUpdate,updateData, reasons) = self.updateProposer.receiveData(data,recall,dcg,userKnown,itemKnown)
        if shouldUpdate:
            d = dict()
            d['Reasons'] = reasons
            d['New entries'] = len(updateData)
            t0 = time.time()
            
            self.updateModel(updateData)
            
            t1 = time.time()
            d['Update time'] = t1-t0
            
            f = open("log/updateItI.log", "a")
            f.write(json.dumps(d)+'\n')
            f.close()
            
    def updateModel(self,data):        
        self.cRows = self.contentRows(self.items)
        self.reverseContentRows = self.reverseDict(self.cRows)
        
        self.M = self.createMatrix()
        
        self.itemRanking = dict()
                    
    def persist(self):
        print("saveMatrices")
        t0 = time.time()

        t1 = time.time()
        print(t1-t0)
        return {'M' : self.M, 'contentRows': self.cRows, 'lastSeen': self.lastSeen}
        
    def restore(self,storedData):
        print("restoreMatrices")
        t0 = time.time()
        
        self.M = storedData['M']
        self.cRows = storedData['contentRows']
        self.lastSeen = storedData['lastSeen']
        
        self.cIds = np.array(list(self.cRows.keys()))
        self.reverseContentRows = self.reverseDict(self.cRows)

        t1 = time.time()
        print(t1-t0)
