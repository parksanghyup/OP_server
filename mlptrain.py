import pandas as pd
import math 
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from MLP import NeuralNetMLP

def get_result_data(rows):

    df  = pd.read_csv('./data/squat_o.csv')
    X = df.iloc[:,1:]
    X = X.values.tolist()  ## DataFrame을 List값으로 추출
    array = []
    for i in range(0,500): ## 좌표값을 math.atan2(y,x) * 180 / math.pi로 두 점 사이의 각도값 추출
        array.append([math.atan2(X[i][5]-X[i][3],X[i][4]-X[i][2]) *180 / math.pi,
                    math.atan2(X[i][7]-X[i][5],X[i][4]-X[i][6]) *180 / math.pi,
                    math.atan2(X[i][9]-X[i][7],X[i][8]-X[i][6]) *180 / math.pi, 1])    
        
    result_o = pd.DataFrame(array) ## List를 DataFrame으로 변환

    df = pd.read_csv('./data/squat_x.csv')
    X = df.iloc[:,1:]
    X = X.values.tolist()
    array = []
    for i in range(0,500): ## 각도값 추출 
        array.append([
                    math.atan2(X[i][5]-X[i][3],X[i][4]-X[i][2]) *180 / math.pi,
                    math.atan2(X[i][7]-X[i][5],X[i][4]-X[i][6]) *180 / math.pi,
                    math.atan2(X[i][9]-X[i][7],X[i][8]-X[i][6]) *180 / math.pi
                    ,0])
        
    result_x = pd.DataFrame(array) ## List를 DataFrame으로 변환

    result_ox = result_o.append(result_x) ## DataFrame 두개를 합쳐서 새로운 DataFrame 정의
    X = result_ox.iloc[0:,:3].values ## 입력 Data 추출
    y = result_ox.iloc[0:,[3]].values  ## 정답 Lable 추출

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        test_size = 0.2,## Dataset을 무작위로 섞어 분리
                                                        random_state=1)

    # print("XTRAIN",type(X_train))
    # print("Xtest",type(X_test))
    # print("ytrain",type(y_train))
    # print("ytest",type(y_test))

    X = { "Head": 0, "Neck": 2, "RShoulder": 4, "RElbow": 6, "RWrist": 8,
      "LShoulder": 10, "LElbow": 12, "LWrist": 14, "Hip": 16, "RHip": 18, 
      "RKnee": 20, "RAnkle": 22, "LHip": 24, "LKnee": 26, "LAnkle": 28, 
      "Background": 30 }

    Y = { "Head": 1, "Neck": 3, "RShoulder": 5, "RElbow": 7, "RWrist": 9,
        "LShoulder": 11, "LElbow": 13, "LWrist": 15, "Hip": 17, "RHip": 19, 
        "RKnee": 21, "RAnkle": 23, "LHip": 25, "LKnee": 27, "LAnkle": 29 }
        
    array = []
    array.append([math.atan2(rows[Y['LHip']]-rows[Y['LShoulder']], 
                            rows[X['LHip']]-rows[X['LShoulder']]) *180 / math.pi,
                math.atan2(rows[Y['LHip']]-rows[Y['LKnee']], 
                            rows[X['LHip']]-rows[X['LKnee']]) *180 / math.pi,
                math.atan2(rows[Y['LAnkle']]-rows[Y['LKnee']], 
                            rows[X['LAnkle']]-rows[X['LKnee']]) *180 / math.pi])    
    result = pd.DataFrame(array)
    A = result.iloc[:,0:3].values


    stdsc = StandardScaler() ## 정규화 객체 생성




    pipe_lr = make_pipeline(StandardScaler(),  ## 순서대로 진행
                        MLPClassifier(hidden_layer_sizes=(100),solver='lbfgs', random_state=3, max_iter=2000))

    pipe_lr.fit(X_train, y_train.ravel())
    # values :  will give the values in an array. (shape: (n,1)
    # ravel() : will convert that array shape to (n, )
    pipe_lr.score(X_test,y_test)

    kfold = StratifiedKFold(n_splits=10,
                       random_state=1).split(X_train,y_train)

    scores = []
    # for k, (train, test) in enumerate(kfold):
    #     pipe_lr.fit(X_train[train], y_train[train])
    #     score = pipe_lr.score(X_train[test], y_train[test])
    #     scores.append(score)
    #     print('폴드: %2d ,정확도 : %.3f' % (k+1, score))

    print('lnCV 정확도: %.3f +/- %.3f' %(np.mean(scores),np.std(scores)))

    print("훈련 정확도:",pipe_lr.score(X_train,y_train.ravel()))
    print("테스트 정확도:",pipe_lr.score(X_test,y_test))
    X_test_pred = pipe_lr.predict(X_test)
    print(X_test_pred)

    print(pipe_lr.predict(A))


    cnt_T =0
    cnt_F =0

    for i in range(0,200):
        if y_test.T[0][i] == 0:
            if y_test.T[0][i] == X_test_pred[i]:
                cnt_F +=1
        if y_test.T[0][i] == 1:
            if y_test.T[0][i] == X_test_pred[i]:
                cnt_T +=1
                
    print("양성 정답 :" ,cnt_T,"음성 정답 :",cnt_F)

    n_epochs =200
    nn = NeuralNetMLP(n_hidden=100, 
                    l2=0, 
                    epochs=n_epochs, 
                    eta=0.0005,
                    minibatch_size=100, 
                    shuffle=True
                    )

    nn.fit(X_train,y_train,X_test,y_test)

    
    print(A)
    return nn.predict(A) # 결과




