import matplotlib.pyplot as plt

# Q1
fig = plt.figure()
ax = plt.axes()

x1s_red = [3, 2, 4, 1]
x2s_red = [4, 2, 4, 4]

# x1s_blue = [2, 4, 4]
# x2s_blue = [1, 3, 1]

ax.plot(x1s_red, x2s_red, 'ro')
# ax.plot(x1s_blue, x2s_blue, 'bo')

# (a)
max_margin_x = [0, 5]
max_margin_y = [-0.5, 4.5]
# ax.plot(max_margin_x, max_margin_y)

# (d)
# sv_plus_x = [0, 5]
# sv_plus_y = [0, 5]
# ax.plot(sv_plus_x, sv_plus_y, 'g--')

# sv_minus_x = [0, 5]
# sv_minus_y = [-1, 4]
# ax.plot(sv_minus_x, sv_minus_y, 'g--')

# (e)
# margin_x = [0, 5]
# margin_y = [-0.7, 4.7]
# ax.plot(max_margin_x, max_margin_y, 'g', alpha=0.45)
# ax.plot(margin_x, margin_y, 'm')

# (g)
x1s_blue = [2, 4, 4, 2]
x2s_blue = [1, 3, 1, 4]
ax.plot(x1s_blue, x2s_blue, 'bo')
ax.plot(max_margin_x, max_margin_y, 'g', alpha=0.10)


ax.set_xlim(xmin=0, xmax=5)
ax.set_ylim(ymin=0, ymax=5)

plt.show()