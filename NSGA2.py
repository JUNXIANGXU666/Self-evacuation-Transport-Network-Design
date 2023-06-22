import numpy as np
import random
import copy
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import math


class SIA():
    def __init__(self,SearchAgents_no,Max_iteration,lb,ub,dim,fobj):
        self.SearchAgents_no=SearchAgents_no
        self.Max_iteration=Max_iteration
        self.lb=lb
        self.ub=ub
        self.dim=dim
        self.fobj=fobj
    def GA(self):       # 遗传算法
        ############ 参数设置####################

        SearchAgents_no=copy.deepcopy(self.SearchAgents_no)
        Max_iteration=copy.deepcopy(self.Max_iteration)
        lb=copy.deepcopy(self.lb)
        ub=copy.deepcopy(self.ub)
        dim=copy.deepcopy(self.dim)
        fobj=copy.deepcopy(self.fobj)

        mut=0.1     # 变异率
        acr=0.7     # 交叉率
        EliminationRatio=0.3    # 淘汰率
        Position=np.zeros((SearchAgents_no,dim))        # 种群个体
        Cost=np.zeros((SearchAgents_no,1))          # 每个个体的适应度
        pop=[Position,Cost]                         # 种群集合

        ############ 初始化种群################
        for i in range(0,SearchAgents_no):
            pop[0][i,:]=(ub-lb)*np.random.rand(dim)+lb
            pop[1][i]=fobj(pop[0][i,:])
        BestSol=[pop[0][0,:],pop[1][0]]                 # 最优个体及适应度
        Convergence_curve=np.zeros((Max_iteration, 1))           # 收敛曲线
        SearchHistory=np.zeros((Max_iteration,dim+1))   # 搜索历史

        ########### 进行迭代寻优###############
        for it in range(0,Max_iteration):
            ## 进行变异和交叉
            for i in range(0,SearchAgents_no):
                # 进行变异
                for j in range(0,dim):
                    if random.random()<=mut:
                        if random.random()<=0.5:
                            pop[0][i,j]=pop[0][i,j]*(1-random.random()*(1-it/Max_iteration)**2)
                        else:
                            pop[0][i,j]=pop[0][i,j]*(1+random.random()*(1-it/Max_iteration)**2)
                # 进行交叉
                if random.random()<acr:
                    acr_var=math.floor((SearchAgents_no-1)*random.random()+1)
                    acr_num=math.floor(((dim-1)*random.random()+1)/2)
                    ordernum=[i for i in range(0,dim)]
                    acr_node=random.sample(ordernum,acr_num)

                    # 开始交叉
                    temp=pop[0][i,acr_node]
                    pop[0][i,acr_node]=pop[0][acr_var,acr_node]
                    pop[0][acr_var,acr_node]=temp

                # 判断是否越界
                Flag4ub=(pop[0][i,:]>ub)+0
                Flag4lb=(pop[0][i,:]<lb)+0
                pop[0][i,:]=pop[0][i,:]*(((Flag4ub+Flag4lb)==0)+0)+ub*(Flag4ub+0)+lb*(Flag4lb+0)

                # 计算适应度
                pop[1][i]=fobj(pop[0][i,:])

            ## 寻找当前追优
            sortorder=np.argsort(pop[1],axis=0).reshape(-1)
            pop[0]=pop[0][sortorder,:]
            pop[1]=pop[1][sortorder]
            if BestSol[1]>pop[1][0]:
                BestSol[0]=pop[0][0,:]
                BestSol[1]=pop[1][0]

            ## 进行优胜劣汰
            temporder=[x for x in range(math.ceil(SearchAgents_no*(1-EliminationRatio)),SearchAgents_no)]
            pop[0][temporder,:]=pop[0][0,:]
            pop[1][temporder]=pop[1][0]

            ## 存放当前最优染色体
            best_score=BestSol[1]
            best_pos=BestSol[0]
            Convergence_curve[it]=best_score
            SearchHistory[it,0:-1]=pop[0][0,:]
            SearchHistory[it,-1]=pop[1][0]

        return best_score, best_pos, Convergence_curve,SearchHistory
    def PSO(self):      # 粒子群算法
        ############## 参数设置 ##############
        SearchAgents_no=copy.deepcopy(self.SearchAgents_no)
        Max_iteration=copy.deepcopy(self.Max_iteration)
        lb=copy.deepcopy(self.lb)
        ub=copy.deepcopy(self.ub)
        dim=copy.deepcopy(self.dim)
        fobj=copy.deepcopy(self.fobj)

        Vmax=6      # 最大速度
        wMax=0.9    # 最大权重
        wMin=0.2    # 最小权重
        c1=2        # 学习因子
        c2=2        # 学习因子

        ############# 种群初始化 ################
        vel=np.zeros((SearchAgents_no,dim))
        pBestScore=np.ones((SearchAgents_no,1))*float('inf')
        pBest=np.zeros((SearchAgents_no,dim))
        gBest=np.zeros((1,dim))
        gBestScore=float('inf')

        # 种群初始化
        pop=[np.zeros((SearchAgents_no,dim)),np.zeros((SearchAgents_no,1))]         # 种群初始化
        for i in range(0,SearchAgents_no):
            pop[0][i,:]=(ub-lb)*np.random.rand(dim)+lb
            pop[1][i]=fobj(pop[0][i,:])

        BestSol=[pop[0][0,:],pop[1][0]]         # 最优个体及适应度
        Convergence_curve=np.zeros((Max_iteration,1))       # 收敛曲线
        SearchHistory=np.zeros((Max_iteration,dim+1))       # 搜索历史

        ############# 开始进行迭代寻优 ####################
        for it in range(0,Max_iteration):
            for i in range(0,SearchAgents_no):
                ## 判断粒子是否越界
                Flag4ub=(pop[0][i,:]>ub)+0
                Flag4lb=(pop[0][i,:]<lb)+0
                pop[0][i,:]=pop[0][i,:]*(((Flag4ub+Flag4lb)==0)+0)+ub*(Flag4ub+0)+lb*(Flag4lb+0)

                # 计算适应度
                fitness=fobj(pop[0][i,:])

                if pBestScore[i]>fitness:
                    pBestScore[i]=fitness
                    pBest[i,:]=pop[0][i,:]
                if gBestScore>fitness:
                    gBestScore=fitness
                    gBest=pop[0][i,:]

            ## 更新粒子
            w=wMax-(it+1)*((wMax-wMin)/Max_iteration)
            for i in range(0,SearchAgents_no):
                for j in range(0,dim):
                    vel[i,j]=w*vel[i,j]+c1*random.random()*(pBest[i,j]-pop[0][i,j])+c2*random.random()*(gBest[j]-pop[0][i,j])

                    # 判断速度是否越界
                    if vel[i,j]>Vmax:
                        vel[i,j]=Vmax
                    if vel[i,j]<-Vmax:
                        vel[i,j]=-Vmax
                    # 更新粒子的位置
                    pop[0][i,j]=pop[0][i,j]+vel[i,j]

            # 记录最优结果
            Convergence_curve[it]=gBestScore
            SearchHistory[it,0:-1]=pop[0][0,:]
            SearchHistory[it,-1]=fobj(pop[0][0,:])

        best_pos=gBest
        best_score=gBestScore
        return best_score, best_pos, Convergence_curve,SearchHistory
    def DE(self):       # 差分进化算法
        ############## 参数设置 ##############
        SearchAgents_no=copy.deepcopy(self.SearchAgents_no)
        Max_iteration=copy.deepcopy(self.Max_iteration)
        lb=copy.deepcopy(self.lb)
        ub=copy.deepcopy(self.ub)
        dim=copy.deepcopy(self.dim)
        fobj=copy.deepcopy(self.fobj)

        beta_min=0.2        # 缩放因子下界
        beta_max=0.8        # 缩放因子上界
        pCR=0.2             # 交叉概率
        pop=[np.zeros((SearchAgents_no,dim)),np.ones((SearchAgents_no,1))*float('inf')]

        for i in range(0,SearchAgents_no):
            pop[0][i,:]=(ub-lb)*np.random.rand(dim)+lb
            pop[1][i]=fobj(pop[0][i,:])

        BestSol=[pop[0][0,:],pop[1][0]]         # 最优个体及适应度
        Convergence_curve=np.zeros((Max_iteration,1))       # 收敛曲线
        SearchHistory=np.zeros((Max_iteration,dim+1))       # 搜索历史

        for it in range(0,Max_iteration):
            for i in range(0,SearchAgents_no):
                x=pop[0][i,:]

                ## 变异操作
                A=[x for x in range(0,SearchAgents_no) if x!=i]
                random.shuffle(A)
                beta=(beta_max-beta_min)*np.random.rand(dim)+beta_min
                y=pop[0][A[0],:]+beta*(pop[0][A[1],:]-pop[0][A[2],:])
                # 判断变异是否越界
                tempmax=np.vstack((y,lb))
                y=tempmax.max(axis=0)
                tempmin = np.vstack((y, ub))
                y=tempmin.min(axis=0)

                ## 交叉操作
                z=np.zeros((1,dim))
                j0=random.randint(0,dim-1)          # 选择交叉位点
                for j in range(0,dim):
                    if j==j0 or random.random()<=pCR:       ## 交叉位点交叉和以一定概率交叉
                        z[0,j]=y[j]
                    else:
                        z[0,j]=x[j]

                ## 优选个体
                NewSol=[z[0,:],fobj(z[0,:])]
                if NewSol[1]<float(pop[1][i]):
                    pop[0][i,:]=NewSol[0]       # 贪婪法选择个体
                    pop[1][i]=NewSol[1]
                    if NewSol[1]<BestSol[1]:
                        BestSol=NewSol

            ## 收集迭代信息
            Convergence_curve[it]=BestSol[1]
            SearchHistory[it,0:-1]=pop[0][0,:]
            SearchHistory[it,-1]=pop[1][0]
        best_score=BestSol[1]
        best_pos=BestSol[0]

        return best_score, best_pos, Convergence_curve,SearchHistory


def PlotCost(pop):
    # 绘制pareto前沿
    nobj = len(pop[0][1])
    nPop = len(pop)
    costdata = np.zeros((nPop, nobj))

    font_tnr = FontProperties(fname=r"C:\WINDOWS\Fonts\times.ttf", size=14)
    font_song = FontProperties(fname=r"C:\WINDOWS\Fonts\simsun.ttc", size=14)
    for i in range(0, nPop):
        for j in range(0, nobj):
            costdata[i, j] = pop[i][1][j]

    for j in range(0, nobj-1):
        for k in range(j+1, nobj):
            fig, ax = plt.subplots()
            ax.scatter(costdata[:, j], -costdata[:, k], s=80, c='b', marker='o')
            # 设置坐标轴标签字体和大小
            # obj1 = 'Objective #' + str(j+1)
            # obj2 = 'Objective #' + str(k+1)
            obj1 = '运输时间'
            obj2 = '可靠性'
            ax.set_xlabel(obj1, fontproperties=font_song)
            ax.set_ylabel(obj2, fontproperties=font_song)
            # 设置刻度字体为数字字体
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontproperties(font_tnr)
            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontproperties(font_tnr)
            plt.show()


def Mutate(x, sigma, VarMin, VarMax):
    # 执行变异操作
    nVar = x.shape[1]
    # 多因子变异
    j = np.random.choice(nVar, np.random.randint(int(nVar/2)))
    # 单因子变异
    j = 0
    y = copy.deepcopy(x)
    y[0, j] = x[0, j] + sigma[0, j] * np.random.randn(1)

    y = np.maximum(y, VarMin)
    y = np.minimum(y, VarMax)

    return y


def Crossover(x1, x2):
    # 执行交叉操作
    alpha = np.random.rand(x1.shape[1])

    y1 = alpha*x1 + (1-alpha)*x2
    y2 = alpha*x2 + (1-alpha)*x1

    return y1, y2


def BinaryTournament(pop):
    # 二元锦标赛
    nPop = len(pop)

    p1 = np.random.randint(0, nPop-1)
    p2 = random.randint(0, nPop-1)
    while p1 == p2:
        p2 = random.randint(0, nPop-1)

    if pop[p1][2] < pop[p2][2]:
        p = pop[p1]
    elif pop[p2][2] < pop[p1][2]:
        p = pop[p2]
    else:
        if pop[p1][5] > pop[p2][5]:
            p = pop[p1]
        else:
            p = pop[p2]
    return p


def SortPopulation(pop):
    temp = np.zeros((2, len(pop)))
    for i in range(0, len(pop)):
        temp[0, i] = pop[i][2]
        temp[1, i] = -pop[i][5]
    SortedIndices = np.lexsort((temp[1, :],temp[0, :]))
    poptemp = copy.deepcopy(pop)
    for i in range(0, len(SortedIndices)):
        poptemp[i] = pop[SortedIndices[i]]

    # 查找所有等级
    R = []
    for i in range(0, len(pop)):
        R.append(pop[i][2])
    nR = max(R)+1

    # 初始化输出
    F = [[] for i in range(0,nR)]

    # 填充输出
    for r in range(0, nR):
        for i in range(0, len(pop)):
            if r == poptemp[i][2]:
                F[r].append(i)
    return poptemp, F


def CalCrowdingDistance(pop, F):
    ## 计算拥挤度
    nF = len(F)
    objnum = len(pop[0][1])

    for k in range(0, nF):
        Nk = len(F[k])

        # 目标1
        Fk = np.zeros((objnum, Nk))
        for i in range(0, Nk):
            for j in range(0, objnum):
                Fk[j,i] = pop[F[k][i]][1][j]
        SortedIndices = np.argsort(Fk[0, :])
        FkSorted = Fk[:, SortedIndices]

        pop[F[k][SortedIndices[0]]][5] = np.inf
        pop[F[k][SortedIndices[-1]]][5] = np.inf

        for i in range(1, Nk-1):
            # if (FkSorted[0, i+1]-FkSorted[0, i-1]) ==0:
            #     pop[F[k][SortedIndices[i]]][5] += 1e-20/ (
            #                 FkSorted[0, -1] - FkSorted[0, 0])
            # else:
            pop[F[k][SortedIndices[i]]][5] += (FkSorted[0, i+1]-FkSorted[0, i-1])/(FkSorted[0, -1]-FkSorted[0,0])

        # 目标2至目标n
        for j in range(1, objnum):
            SortedIndices = np.argsort(Fk[j,:])
            FkSorted = Fk[:, SortedIndices]

            pop[F[k][SortedIndices[0]]][5] = np.inf
            pop[F[k][SortedIndices[-1]]][5] = np.inf

            for i in range(1, Nk-1):
                pop[F[k][SortedIndices[i]]][5] += (FkSorted[j, i + 1] - FkSorted[j, i - 1]) / (
                            FkSorted[j, -1] - FkSorted[j, 0])
    return pop


def NonDominatedSorting(pop):
    # 个体数量
    nPop = len(pop)

    # 非支配排序
    F = [[]]
    for i in range(0, nPop):
        pop[i][4] = 0
        pop[i][3] = []
        for j in range(0, nPop):
            if Dominates(pop[i], pop[j]):
                pop[i][4] = pop[i][4] + 1
            elif Dominates(pop[j], pop[i]):
                pop[i][3].append(j)
        if pop[i][4] == 0:
            F[0].append(i)

    k = 0
    while True:
        temp = []
        for i in F[k]:
            for j in pop[i][3]:
                pop[j][4] = pop[j][4] - 1
                if pop[j][4] == 0:
                    temp.append(j)
        if len(temp) == 0:
            break
        F.append(temp)
        k = k + 1

    for i in range(0, len(F)):
        for j in F[i]:
            pop[j][2] = i
    return pop, F


def Dominates(popx, popy):
    num=len(popx[1])
    temp1 = all(popx[1][t] <= popy[1][t] for t in range(0,num))
    temp2 = any(popx[1][t] < popy[1][t] for t in range(0,num))
    return (temp1 and temp2)


def costfunction(var):
    global topology, transp_time
    # print(topology)
    # print(transp_time)

    nNum = topology.shape[0]

    topologytemp = copy.deepcopy(topology)
    count = 0
    for i in range(0, nNum):
        for j in range(i+1, nNum):
            if topologytemp[i, j] == 0:
                topologytemp[i, j] = np.round(var[0, count])
                topologytemp[j, i] = np.round(var[0, count])
                count = count + 1

    # 总运行时间
    total_transportation_time = 0
    for i in range(0, nNum):
        for j in range(i+1, nNum):
            if topologytemp[i, j] == 1:
                total_transportation_time = total_transportation_time + transp_time[i, j]

    # 可靠度计算
    reliability = 0
    Average_node_degree = np.sum(topologytemp)/nNum
    for i in range(0, nNum):
        if i == 0:
            reliability = np.sum(topologytemp[:, i])/Average_node_degree
        reliability = reliability * np.sum(topologytemp[:, i])/Average_node_degree

    # var = var[0]
    # result1 = var[0]
    # num = len(var)
    # g = 1 + 9/(num - 1)*np.sum(var[1:-1])
    # h = 1 - (result1 / g)**2
    # result2 = g*h
    return [total_transportation_time, -reliability]


def costfunctionother(var):
    global topology, transp_time
    # print(topology)
    # print(transp_time)

    nNum = topology.shape[0]

    topologytemp = copy.deepcopy(topology)
    count = 0
    for i in range(0, nNum):
        for j in range(i+1, nNum):
            if topologytemp[i, j] == 0:
                topologytemp[i, j] = np.round(var[count])
                topologytemp[j, i] = np.round(var[count])
                count = count + 1

    # 总运行时间
    total_transportation_time = 0
    for i in range(0, nNum):
        for j in range(i+1, nNum):
            if topologytemp[i, j] == 1:
                total_transportation_time = total_transportation_time + transp_time[i, j]

    # 可靠度计算
    reliability = 0
    Average_node_degree = np.sum(topologytemp)/nNum
    for i in range(0, nNum):
        if i == 0:
            reliability = np.sum(topologytemp[:, i])/Average_node_degree
        reliability = reliability * np.sum(topologytemp[:, i])/Average_node_degree

    result = total_transportation_time/300.0 - reliability

    return result

def main():
    ## 导入数据
    topologydata = pd.read_excel('./data.xlsx', sheet_name='拓扑图')
    transptimedata = pd.read_excel('./data.xlsx', sheet_name='运输时间')

    ## 定义全局变量
    global topology, transp_time
    topology = np.array(topologydata.iloc[:, 1:])
    transp_time = np.array(transptimedata.iloc[:, 1:])

    fobj = costfunction
    nVar = 181
    VarMin = np.ones((1, nVar)) * 0
    VarMax = np.ones((1, nVar)) * 1

    # NSGA2参数
    Maxlt = 100  # 最大迭代次数
    nPop = 100  # 种群个数
    pc = 0.7  # 交叉率
    nc = 2 * np.round(pc * nPop / 2)  # 子代数量（父代）
    pm = 0.3  # 变异率
    nm = np.round(pm * nPop)  # 变异种群个数
    mu = 0.02  # 变异比率
    sigma = 0.1 * (VarMax - VarMin)  # 变异步长

    NSGA_convergence_curve = np.zeros((Maxlt, 2))
    # 初始化操作
    empty_individual = [[], [], [], [], [], 0]
    # Position,Cost,Rank,DominationSet,DominatedCount,CrowdingDistance
    pop = []
    for i in range(0, nPop):
        pop.append(copy.deepcopy(empty_individual))

    for i in range(0, nPop):
        for j in range(0, nVar):
            pop[i][0] = np.random.rand(1,nVar) * (VarMax - VarMin) + VarMin
        pop[i][1] = fobj(pop[i][0])

    ## 进入迭代
    print("\n<<<<<<<<<<<<<<<<<开始NSGAⅡ迭代<<<<<<<<<<<<<<<<<<<<<<")
    for it in range(0, Maxlt):
        # 非支配排序
        pop, F = NonDominatedSorting(pop)
        # 计算拥挤度
        pop = CalCrowdingDistance(pop, F)
        # 种群排序
        pop, F = SortPopulation(pop)

        # 展示迭代次数及支配等级
        print('迭代次数', it+1, ':Pareto前沿数量=', len(F))

        # 生成子代
        popc = [[], []]
        for i in range(0, int(nc/2)):
            popc[0].append(copy.deepcopy(empty_individual))
            popc[1].append(copy.deepcopy(empty_individual))
        for k in range(0, int(nc/2)):
            # 选择父代
            p1 = BinaryTournament(pop)
            p2 = BinaryTournament(pop)

            # 执行交叉
            popc[0][k][0], popc[1][k][0] = Crossover(p1[0], p2[0])

            # 执行变异
            popc[0][k][0] = Mutate(popc[0][k][0], sigma, VarMin, VarMax)
            popc[1][k][0] = Mutate(popc[1][k][0], sigma, VarMin, VarMax)

            # 评估父代
            popc[0][k][1] = fobj(popc[0][k][0])
            popc[1][k][1] = fobj(popc[1][k][0])

        popctemp = []
        for i in range(0,len(popc[0])):
            popctemp.append(popc[0][i])
            popctemp.append(popc[1][i])

        for i in range(0, len(popctemp)):
            pop.append(popctemp[i])

        # 非支配排序
        pop, F = NonDominatedSorting(pop)

        # 计算拥挤度
        pop = CalCrowdingDistance(pop, F)

        # 种群排序
        pop, F = SortPopulation(pop)

        # 筛选最优种群
        pop = pop[0:nPop]

        # 第一个种群的变化曲线
        NSGA_convergence_curve[it, 0] = pop[0][1][0]
        NSGA_convergence_curve[it, 1] = pop[0][1][1]

    ## 计算结果

    # 打印非支配解
    print('非支配解')
    countnum = 1
    for i in range(0, nPop):
        if pop[i][2] == 0:
            print('第', countnum, '解')

            nNum = topology.shape[0]
            var = pop[i][0]
            topologytemp = copy.deepcopy(topology)
            count = 0
            for k in range(0, nNum):
                for j in range(k + 1, nNum):
                    if topologytemp[k, j] == 0:
                        topologytemp[k, j] = np.round(var[0, count])
                        topologytemp[j, k] = np.round(var[0, count])
                        count = count + 1

            print('x=', topologytemp)
            print('f=', pop[i][1])
            countnum += 1

    ## 以下是算法对比
    print('<<<<<<<<<<<<<<<<<<遗传算法求解<<<<<<<<<<<<<<<<<<<<<<<')
    fobj = costfunctionother
    ObjCompare = SIA(nPop, Maxlt, VarMin, VarMax, nVar, fobj)
    best_score, best_pos, GA_Convergence_curve, SearchHistory = ObjCompare.GA()
    nNum = topology.shape[0]
    topologytemp = copy.deepcopy(topology)
    count = 0
    for k in range(0, nNum):
        for j in range(k + 1, nNum):
            if topologytemp[k, j] == 0:
                topologytemp[k, j] = np.round(best_pos[count])
                topologytemp[j, k] = np.round(best_pos[count])
                count = count + 1

    print('GA:x=', topologytemp)
    print('GA:f=', best_score)

    print('<<<<<<<<<<<<<<<<<<粒子群算法求解<<<<<<<<<<<<<<<<<<<<<<<')
    best_score, best_pos, PSO_Convergence_curve, SearchHistory = ObjCompare.PSO()
    nNum = topology.shape[0]
    topologytemp = copy.deepcopy(topology)
    count = 0
    for k in range(0, nNum):
        for j in range(k + 1, nNum):
            if topologytemp[k, j] == 0:
                topologytemp[k, j] = np.round(best_pos[count])
                topologytemp[j, k] = np.round(best_pos[count])
                count = count + 1
    print('PSO:x=', topologytemp)
    print('PSO:f=', best_score)

    print('<<<<<<<<<<<<<<<<<<进化差分算法求解<<<<<<<<<<<<<<<<<<<<<<<')
    best_score, best_pos, DE_Convergence_curve, SearchHistory = ObjCompare.DE()
    nNum = topology.shape[0]
    topologytemp = copy.deepcopy(topology)
    count = 0
    for k in range(0, nNum):
        for j in range(k + 1, nNum):
            if topologytemp[k, j] == 0:
                topologytemp[k, j] = np.round(best_pos[count])
                topologytemp[j, k] = np.round(best_pos[count])
                count = count + 1
    print('DE:x=', topologytemp)
    print('DE:f=', best_score)

    # 绘制pareto解
    PlotCost(pop)

    # 打印迭代曲线
    font_song = FontProperties(fname=r"C:\WINDOWS\Fonts\simsun.ttc", size=14)
    font_tnr = FontProperties(fname=r"C:\WINDOWS\Fonts\times.ttf", size=14)
    fig, ax = plt.subplots()
    ax.plot(NSGA_convergence_curve[:, 0], label='NSGA')
    # 设置坐标轴标签字体和大小
    ax.set_xlabel("迭代次数/次", fontproperties=font_song)
    ax.set_ylabel("适应度#1", fontproperties=font_song)
    # 设置刻度字体为数字字体
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontproperties(font_tnr)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontproperties(font_tnr)
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(GA_Convergence_curve, label='GA')
    ax.plot(PSO_Convergence_curve, label='PSO')
    ax.plot(DE_Convergence_curve, label='DE')
    ax.legend()
    # 设置坐标轴标签字体和大小
    ax.set_xlabel("迭代次数/次", fontproperties=font_song)
    ax.set_ylabel("适应度", fontproperties=font_song)
    # 设置刻度字体为数字字体
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontproperties(font_tnr)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontproperties(font_tnr)
    plt.show()




main()
