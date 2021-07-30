import pandas as pd
import time
import numpy as np
import math
import pika
import json
from bson import json_util
import sys

class EuclideanItemRecommender:
    def __init__(self,d,l):
        self.itemRanking = dict()
        self.lastSeen = dict()
        self.items = dict()
        self.itemsSeen = dict()
        self.d = d
        self.l = l
    
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
    
    def setItemsSeen(self,user,item):
        if item not in self.itemsSeen:
            self.itemsSeen[item] = set()
        self.itemsSeen[item].add(user)  
        
    def newItem(self,item):
        self.items[item] = np.random.rand(self.d)
        
    def P(self,j,i):
        num = np.exp(-np.linalg.norm(self.items[j]-self.items[i]))
        den = sum([np.exp(-np.linalg.norm(self.items[l]-self.items[i])) for l in self.items if (l != i)])
        return num/den
    
    def calculateDiffI(self,i,j,k):
        diff = self.items[j][k] - self.items[i][k]
        num = sum([np.exp(-np.linalg.norm(self.items[l]-self.items[i]))*(self.items[l][k]-self.items[i][k]) for l in self.items if (l != i)])
        den = sum([np.exp(-np.linalg.norm(self.items[l]-self.items[i])) for l in self.items if (l != i)])
        result = 2*(diff-num/den)
        return result
    
    def calculateDiffJ(self,i,j,k):
        diff = self.items[j][k] - self.items[i][k]
        p = self.P(j,i)
        result = -2*diff*(1-p)
        return result
    
    def processData(self,user,item):
        self.setItemsSeen(user,item)
        if item not in self.items:
            self.newItem(item)
        if user in self.lastSeen:
            last = self.lastSeen[user]
            self.lastSeen[user] = item     
            self.improve(item,last)
        else:
            self.lastSeen[user] = item
    
    def trainItems(self,data):
        for r in zip(data['ownerId'], data['contentId']):
            self.processData(r[0],r[1])
            
    def createMatrix(self):
        print("Creating matrix")
        t0 = time.time()
    
        l = len(self.items)
        M = np.zeros((l,l))

        coords = np.array(np.meshgrid(range(l),range(l))).T.reshape(-1,2)
        for i,j in coords:
            M[i,j] = self.P(self.reverseContentRows[j],self.reverseContentRows[i])
                
        t1 = time.time()
        print(t1-t0)
            
        return M
               
    def initializeModel(self,ratings):
        t0 = time.time()
          
        self.cIds = self.contentIds(ratings)      
        self.cRows = self.contentRows(self.cIds)
        self.reverseContentRows = self.reverseDict(self.cRows)
        
        self.trainItems(ratings)
        self.M = self.createMatrix()

        t1 = time.time()
        print("Training time: " + str(t1-t0))
        
    def improve(self,j,i):
        for k in range(len(i)):
            diffI = self.calculateDiffI(i,j,k)
            diffJ = self.calculateDiffJ(i,j,k)
            self.items[i][k] = self.items[i][k] + self.l*diffI
            self.items[j][k] = self.items[j][k] + self.l*diffJ
        
    def getUserRank(self,userId):
        if userId not in self.lastSeen:
            return (False, [])
        else:
            return self.getItemRank(self.lastSeen[userId])
        
    def userTop(self, uId,n):
        userRanking = self.getUserRank(uId)
        return (userRanking[0],userRanking[1][0:n])
        
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
                itemRanking.append((cId, self.M[self.cRows[itemId],self.cRows[cId]]))
        
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

                f = open("log/updateEIR.log", "a")
                f.write(json.dumps(d)+'\n')
                f.close()
            
    def updateModel(self,data):        
        for d in data:
            user = d['ownerId']
            item = d['contentId']
            self.processData(user,item)
        
        self.M = self.createMatrix()
        
        self.itemRanking = dict()
        
    def MPR(self,baseItem,seenItem,n):
        rec = self.getItemRank(baseItem,n)
        i = 0
        index = n
        for r in rec[1]:
            if r[0] == seenItem:
                index = i
                break
            i += 1
        if index == n:
            return (1,-1)
        else:
            num = sum([i/n*len(self.itemsSeen[rec[1][i][0]]) for i in range(index)])
            num += index/n*len(self.itemsSeen[rec[1][index][0]])/2
            den = sum([len(self.itemsSeen[r[0]]) for r in rec[1]])
            return (num/den,index)
                    
    def persist(self):
        print("saveMatrices")
        t0 = time.time()

        t1 = time.time()
        print(t1-t0)
        return {'lastSeen': self.lastSeen, 'items': self.items, 'M': self.M, 'itemsSeen': self.itemsSeen, 'contentRows': self.cRows}
        
    def restore(self,storedData):
        print("restoreMatrices")
        t0 = time.time()
        
        self.lastSeen = loaded['lastSeen']
        self.items = loaded['items']
        self.M = loaded['M']
        self.itemsSeen = loaded['itemsSeen']
        self.cRows = loaded['contentRows']
        
        self.cIds = np.array(list(self.cRows.keys()))
        self.reverseContentRows = self.reverseDict(self.cRows)

        t1 = time.time()
        print(t1-t0)
