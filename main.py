# -*- coding:utf-8 -*-
#coding: unicode_escape
import pandas as pd
import math

trainLable={}
testLable={}
features=[]
ftmap={}

# def getFeatNum():
class btree:
    def __init__(self,subft,feature):
        self.subft=subft
        self.feature=feature
    subft={}
    feature=''


def myEntropy(lables):
    good=0
    bad=0
    entropy=0
    total=len(lables)
    for l in lables.values():
        if l=="是":
            good+=1
        else:
            bad+=1
    if good!=0:
        entropy+=-good/total*math.log2(good/total)
    if bad!=0:
        entropy+=-bad/total*math.log2(bad/total)
    return entropy

def myConditionalEntropy(ft):
    ft_p=[]
    total=0
    entropy=0
    for i in ft:
        ft_p.append(len(ft[i]))
        total+=len(ft[i])
    key=list(ft.keys())
    for i in range(len(ft_p)):
        ft_p[i]=ft_p[i]/total
        tempdict={}
        for j in ft[key[i]]:
            tempdict[j[0]]=j[1]
        entropy+=ft_p[i]*myEntropy(tempdict)
    return entropy

def sortTrain(train):
    index=0
    for i in range(1,len(train[0])-1): # 第一个是序号，最后一个是标签所以都不用审核
        if type(train[0][i])==float:
            index=i
            break
    order=[]
    for i in train:
        order.append(i[index])
    order=sorted(order)
    newtrain=[]
    for i in order:
        for j in train:
            if j[index]==i:
                newtrain.append(j)
    return newtrain

def continuousConentr(train,entro):
    """
    :param train:
    :param entro: 总的HD
    :return:
    因为连续值计算肯定用了c4.5所以在这里面就直接算了c4.5的大小
    """
    maxvalue=0
    ratio=0
    # 下面这个循环是先找到哪个是连续值
    for i in range(1,len(train[0])-1): # 第一个是序号，最后一个是标签所以都不用审核
        if type(train[0][i])==float:
            index=i
            break
    # 下面是把分成两个集合
    for i in range(1,len(train)-1):
        t=0
        l1=[]
        l2=[]
        while t<len(train):
            if t<i:
                l1.append([train[t][0],train[t][len(train[t])-1]])
            else:
                l2.append([train[t][0],train[t][len(train[t])-1]])
            t+=1
        dic={}
        dic["less"]=l1
        dic["more"]=l2
        myentro=myConditionalEntropy(dic)
        total=len(train)
        had=-(i/total*math.log2(i/total)+(total-i)/total*math.log2((total-i)/total))
        re=(entro-myentro)/had
        if re>ratio:
            ratio=re
            maxvalue=train[i][index]
        return ratio,maxvalue

def had(split):
    num=[]
    for k in split:
        num.append(len(split[k]))
    s=sum(num)
    re=0
    for i in num:
        re+=i/s*math.log2(i/s)
    return -re

def id3Train(train,lable,nodes,ifcontinuous=False,depth=0):
    # End condition: every tribute has the same class
    if ifcontinuous==True:
        train=sortTrain(train)
    elenum=len(train[0])
    tc= list(lable.values())
    ifsame=1
    for i in range(len(tc)-1):
        if tc[i] != tc[i+1]:
            ifsame = 0
            break
    if ifsame==1:
        nodes.feature=tc[0]
        nodes.subft=train
        return nodes
    if len(train[0])<=2:
        nodes.feature=tc[0]
        nodes.subft=train
        return nodes
    if len(train)<=1:  # 退出条件，剪枝条件，节点中样本树
        isgood=0
        notgood=0
        for i in train:
            if i[len(i)-1]=="是":
                isgood+=1
            else:
                notgood+=1
        if isgood>notgood:
            nodes.feature="是"
        else:
            nodes.feature="否"
        nodes.subft=train
        return nodes
    if depth>=2:  # 剪枝方式，限制最大深度
        isgood = 0
        notgood = 0
        for i in train:
            if i[len(i) - 1] == "是":
                isgood += 1
            else:
                notgood += 1
        if isgood > notgood:
            nodes.feature = "是"
        else:
            nodes.feature = "否"
        nodes.subft = train
        return nodes
    # 进入下一次递归
    else:
        depth+=1
        entro=myEntropy(lable)
        print("entropy of H(D) is ",entro)
        chosenNum=1
        chosen=1000
        splitValue=0
        ratio=0
        for i in range(1,elenum-1): # 遍历每个feature
            splitByFeat={}
            if type(train[0][i])!=float: # 离散值
                for j in train:
                    if j[i] in splitByFeat.keys():
                        splitByFeat[j[i]].append([j[0],j[elenum-1]]) # 如果已经有这个键，就在这个键对应的列表上添加这个样本的【编号，是否好】
                    else:
                        splitByFeat[j[i]]=[[j[0],j[elenum-1]]]
                conEntr = myConditionalEntropy(splitByFeat)
                gain = entro - conEntr
                if gain==0:
                    continue
                if ifcontinuous==True:
                    tratio = gain / had(splitByFeat)
            else:  #连续值
                tratio,splitValue=continuousConentr(train,entro)
            # 这个是基础要求中的条件
            if conEntr<chosen and ifcontinuous==False:
                chosen=conEntr
                chosenNum=i
            # 这个是中级要求条件
            elif ifcontinuous==True:
                if tratio > ratio:
                    ratio=tratio
                    chosenNum=i
            print(splitByFeat)
            print("conditional entropy of H(D|x",i,")is",conEntr)
        print("finnaly we gained most entropy by chosing the feature",chosenNum)

        newNode=features[ftmap[train[0][chosenNum]]]
        childtree={}
        if newNode!="密度":
            nodes.feature = newNode  #如果本身就是离散值，就把这个属性名字作为节点名字
            for i in train:
                temp=i[chosenNum]
                if i[chosenNum] in list(childtree.keys()):
                    del i[chosenNum]
                    childtree[temp].append(i)
                else:
                    del i[chosenNum]
                    childtree[temp]=[i]
        else:
            nodes.feature = splitValue
            childtree['<=']=[]
            childtree['>']=[]
            for i in train:
                if i[chosenNum]<=splitValue:  # 如果连续值，就用分裂点的值作为其节点值
                    del i[chosenNum]
                    childtree['<='].append(i)
                else:
                    del i[chosenNum]
                    childtree['>'].append(i)
        for key in childtree:
            nodes.subft[key]=btree({},'')
            templable={}
            for i in childtree[key]:
                templable[i[0]] = i[len(i) - 1]
            id3Train(childtree[key],templable,nodes.subft[key],ifcontinuous,depth)

    return nodes
def printTree(tree,depth):
    space="   "
    if tree.feature=="是" or tree.feature=="否":
        print(depth*space,tree.feature)
        return
    else:
        print(space*depth,tree.feature)
        depth+=1
        for i in list(tree.subft.values()):
            printTree(i,depth)
        return

def predict(test,node):
    while node.feature!="是" and node.feature!="否":
        if type(node.feature)==float:
            if test[5]<=node.feature:
                node=node.subft["<="]
            else:
                node=node.subft[">"]
        else:
            index=features.index(node.feature)
            node=node.subft[test[index]]
    return node.feature

def loaddata(trainpath,testpath):
    global features
    train=pd.read_csv(trainpath,encoding='gbk')
    features=[col for col in train]
    for i in range(1,len(features)-1):
        for j in train[features[i]]:
            ftmap[j]=i
    test=pd.read_csv(testpath,encoding='gbk')
    train=train.values.tolist()
    test=test.values.tolist()
    for i in train:
        trainLable[i[0]]=i[len(i)-1]
    for i in test:
        testLable[i[0]]=i[len(i)-1]
    return train,test


if __name__=="__main__":
    trainpath1="Watermelon-train1.csv"
    testpath1="Watermelon-test1.csv"
    train,test=loaddata(trainpath1,testpath1)
    # 基础要求
    treeRoot=btree({},'')
    treeRoot=id3Train(train,trainLable,treeRoot)
    printTree(treeRoot,0)
    rightpre=0
    for i in test:
        re=predict(i,treeRoot)
        if re==i[len(i)-1]:
            rightpre+=1
            print(i,"is correctly predicted")
    print("accuracy is",rightpre/len(test))

    # 用c4.5的方法
    trainpath2 = "Watermelon-train2.csv"
    testpath2 = "Watermelon-test2.csv"
    train,test=loaddata(trainpath2,testpath2)
    treeRoot2=btree({},'')
    treeRoot2=treeRoot=id3Train(train,trainLable,treeRoot2,ifcontinuous=True)
    printTree(treeRoot2,0)
    rightpre=0
    for i in test:
        re=predict(i,treeRoot2)
        if re==i[len(i)-1]:
            rightpre+=1
            print(i,"is correctly predicted")
    print("accuracy is",rightpre/len(test))


