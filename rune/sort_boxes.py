boxes=[(22,10,5),(30,11,6),(50,9,7),(20,40,8),(32,43,9),(52,43,1),(21,1,2),(33,3,3),(54,0,4)]
def getKey(item):
    return item[1]
def find_top(boxes):
    sth=[]
    while(len(sth)<3):
        m=max(boxes,key=getKey)
        sth.append(m)
        boxes.remove(m)
    return sth
def find_second(boxes):
    sth=[]
    while(len(sth)<3):
        m=max(boxes,key=getKey)
        sth.append(m)
        boxes.remove(m)
    return sth
if __name__=="__main__":
    boxes=[(22,10,5),(30,11,6),(50,9,7),(20,40,8),(32,43,9),(52,43,1),(21,1,2),(33,3,3),(54,0,4)]
    top=find_top(boxes)
    second=find_second(boxes)
    remain=boxes
    top=sorted(top)
    second=sorted(second)
    remain=sorted(remain)
    print(top)
    print(second)
    print(remain)
