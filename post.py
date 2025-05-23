import cv2

# 1. Load your image
img = cv2.imread('C:/Users/mghir/desktop/test1.jpg')
clone = img.copy()

# 2. List to store click coordinates
coords = []

# 3. Mouse callback that records (x,y) on left‑click
def click_event(event, x, y, flags, param):
    global coords, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append((x, y))
        # draw a small circle so you can see the click
        cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("image", clone)
        print(f"Point recorded: {(x, y)}")

# 4. Show window and bind callback
cv2.imshow("image", clone)
cv2.setMouseCallback("image", click_event)

print("Click on the four corners of each field in order: top‑left, top‑right, bottom‑right, bottom‑left.")
print("Press 'q' when done.")

# 5. Wait for user to press 'q'
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# 6. Now `coords` holds the list of clicked points.
#    e.g. coords = [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
#    To make a bounding box: x_min = min(x’s), y_min = min(y’s), etc.
if coords:
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(xs), max(ys)
    print(f"Derived box = ({x0}, {y0}, {x1}, {y1})")
