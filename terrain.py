import cv2
import numpy as np

perlin = cv2.imread('perlin.png', cv2.IMREAD_GRAYSCALE)

#result = perlin.copy()
#cv2.imshow('result', result)

landLower, landUpper = 0, 100
waterLower, waterUpper = 100, 256

#landColor = [234,198,56]
landColor = [37,206,0]
waterColor = [255,101,0]

result = np.full((512, 512, 3), 0, dtype=np.uint8)
landBgr = np.full((512, 512, 3), landColor, dtype=np.uint8)
waterBgr = np.full((512, 512, 3), waterColor, dtype=np.uint8)

map_land = cv2.inRange(perlin, landLower, landUpper)
map_water = cv2.inRange(perlin, waterLower, waterUpper)

result = np.where(map_land[:, :, None], landBgr, result)
result = np.where(map_water[:, :, None], waterBgr, result)
#landBgr.copyTo(result, map_land)
cv2.imshow('result', result)

cv2.waitKey(0)