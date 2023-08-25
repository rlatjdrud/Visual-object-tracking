import numpy as np
import math

def _create_anchor(response_sz,ratio,scale,stride):
    #response_sz : backbone을 통과한 후 feature map의 크기
    #ratio : 앵커박스의 가로 세로 비율
    #scale : 앵커박스의 크기
    #stride : response_sz가 만들 때 사용한 stride

    anchor_num = len(ratio)*len(scale) 
    #ratio가 3개이고, scale도 3개이면 총 9개의 서로다른 앵커박스를 만든다는 뜻
    center_num = stride*stride
    anchors = np.zeros((anchor_num,4),np.float32)
    # 9x4로 한 세트의 앵커박스를 뼈대 구축

    ind=0
    for r in ratio : 
        w = int(math.sqrt(center_num/r))
        h = int(w*r)
        #한 ratio에 대한 3개의 scale의 w,h를 구한다.

        for s in scale :
            anchors[ind,0] = 0
            anchors[ind,1] = 0
            anchors[ind,2] = w*s
            anchors[ind,3] = h*s
            ind+=1
            #--------------------하나의 ratio에 대해서 scale 3개의 for문을 완료하면 9x4에서 0~2행의 w,h가 채워진다.------------------------#
            #--------------------3개의 ratio를 다 수행하면 9x4의 w,h가 채워진다.
            
 
    anchors = np.tile(anchors,response_sz*response_sz).reshape(-1,4)
    #9x4행렬을 한 원소로 가정하고 이 원소를 19*19=(361,) 형태로 복사한다. 즉, 9x4*361 = 9x 1444 가 된다. 
    #이 행렬의 0행의 구성은 [0 0 w1*s1 h1*s1]이 총 361개가 있는것이다.
    # 1행의 구성은 [0 0 w1*s2,h1*s2 ]가 총 361개가 있다.
    # 2행의 구성은 [0 0 w1*s3,h1*s3 ]가 총 361개가 있다.
    # 이런 방식으로 구성된다.
    # 이를 (-1,4)로 리쉐입하면 
    # 3249x4가 된다.이 행렬의 굿성은 일단 원래 행열은 0행에 [0 0 w1*s1,h1*s1 ]가 총 361개로 구성되어있었다. 이 값이 3249x4행렬의 0~360행을 채운다. 
    # 즉 3249x4행렬의 360행렬까지는 하나의 w1*s1 h1*s1을 가진다. 
    # 361~720행까지는 하나의 w1*s2 h1*s2을 가진다.
    begin = -(response_sz//2)*stride
    #-72

    x = np.arange(response_sz)*8+begin
    # 0 1 2 ---- 18 -> 0 8 16 ----- 144 -> -72 -64 ---0 ---- 64 72
    y = np.arange(response_sz)*8+begin
    # 0 1 2 ---- 18 -> 0 8 16 ----- 144 -> -72 -64 ---0 ---- 64 72

    x_grid,y_grid = np.meshgrid(x,y)
    #x_grid는 19x19 행렬이고, 각 행에 -72 -64 ---0 ---- 64 72으로 구성된다.
    #ㅛ_grid는 19x19 행렬이고, 각 열에 -72 -64 ---0 ---- 64 72으로 구성된다.
    x_grid = np.tile(x_grid.flatten(),(anchor_num,1)).flatten()
    y_grid = np.tile(y_grid.flatten(),(anchor_num,1)).flatten()
    #x_grid.flatten()하게되면 |-72 -64 ---0 ---- 64 72|를 한 세트로 보면 총 이 세트가 19개가 일렬로 늘여선다. 이를 한 행을 총 9행으로 복사한다.

    #|-72 -64 ---0 ---- 64 72| ------ |-72 -64 ---0 ---- 64 72| 19*19 = 361개 #
    #|-72 -64 ---0 ---- 64 72| ------ |-72 -64 ---0 ---- 64 72| #
    #|-72 -64 ---0 ---- 64 72| ------ |-72 -64 ---0 ---- 64 72| #
    # -------------------------------------------------------
    #|-72 -64 ---0 ---- 64 72| ------ |-72 -64 ---0 ---- 64 72| #
                            #9x361#

    #이 행렬을 또 flatten한다. 즉 |-72 -64 ---0 ---- 64 72| ------ |-72 -64 ---0 ---- 64 72|가 한 세트로 보면 이 세트가 9개가 일렬로 늘어선다. (3249, )

    #y_grid
    #|-72 -72 --- -72 ---- -72| --|0 0 ---0---- 0 0|--- |72 72 ---72 ---- 72 72| (3249,)
   
    
    anchors[:,0]=x_grid
    # (3249, )를 3249x4의 모둔 0열에 넣는다. 
    anchors[:,1]=y_grid
    # (3249, )를 3249x4의 모둔 1열에 넣는다. 
    return anchors
    #|-72 -72 w1*s1 w1*s1|#
    #|-64 -72 w1*s1 w1*s1|#
    #|-56 -72 w1*s1 w1*s1|#
    #---------------------#
    #|56 -72 w1*s1 w1*s1|#
    #|64 -72 w1*s1 w1*s1|#
    #|72 -72 w1*s1 w1*s1|#
    #anchors 3249x4행렬에서 0~18행렬의 구성이다.




if __name__ == "__main__":
    anchors=_create_anchor(19,[0.5,1,2],[0.5,1,2],8)
    print(anchors[:19])
    