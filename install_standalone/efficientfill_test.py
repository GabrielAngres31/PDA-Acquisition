import types
import numpy as np

def MyFill(array:np.array, x:int, y:int):
    if not array[y][x]:
        _MyFill(array, x, y, array.shape[1], array.shape[0])

def _MyFill(array:np.array, x:int, y:int, width:int, height:int):
    while(True):
        ox = x 
        oy = y
        while(y != 0 and not array[y-1, x]): y -= 1
        while(x != 0 and not array[y, x-1]): x -= 1 
        if(x == ox and y == oy): break
    MyFillCore(array, x, y, array.shape[1], array.shape[0])

def MyFillCore(array:np.array, x:int, y:int, width:int, height:int):
    lastRowLength = 0
    while(True):

        rowLength = 0
        sx = x
        if(lastRowLength != 0 and array[y, x]):
            while(True):
                x += 1
                lastRowLength -= 1
                if lastRowLength == 0:
                    return
                if not array[y, x+1]:
                    break   
            sx = x
        else:
            print(array[y,x])
            while(x != 0 and not array[y,x-1]):
                print(x)
                rowLength += 1
                lastRowLength += 1
                x -= 1
                array[y][x] = 1
                if y != 0 and not array[y-1, x]:
                    _MyFill(array, x, y-1, width, height)
        while(sx < width and not array[y, sx]):
            rowLength += 1
            sx += 1
            array[y][sx] = 1

        if rowLength < lastRowLength:
            while(sx + 1 < x+lastRowLength):
                sx += 1
                if not array[y][sx]:
                    MyFillCore(array, sx, y, width, height)
        elif rowLength > lastRowLength and y != 0:
            ux = x+lastRowLength
            while(ux + 1 < sx):
                ux += 1
                if not array[y-1][ux]:
                    _MyFill(array, ux, y-1, width, height)
        lastRowLength = rowLength
        
        if lastRowLength == 0 or y+1 < height:
            break
        else:
            y += 1
        

array_test = np.array([
                [0,0,0,0,0,1,0,0],
                [0,1,1,1,1,1,0,0],
                [0,1,0,0,0,1,0,0],
                [0,1,1,0,0,1,0,0],
                [0,0,0,1,0,1,0,0],
                [0,0,0,1,1,1,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,1,0,0]])
print(array_test)

MyFill(array_test, 3, 2)

print(array_test)