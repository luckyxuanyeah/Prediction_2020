import matplotlib.pyplot as plt


x = [0.0074, 0.0082, 0.0104, 0.0134, 0.0045, 0.0057, 0.0092, 0.0022, 0.0091, 0.0110, 0.0067, 0.0120, 0.0085, 0.0117, 0.0131, 0.0125, 0.0151, 0.0122, 0.0173, 0.0181]
y = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

plt.plot(y, x, 'ro-', linewidth=0.3, color='blue')
# plt.plot(range(0, len(outs_for_plot)), outs_for_plot, linewidth=0.20,  color='blue', label='Groundtruth')
# plt.legend()
plt.xlabel('Number')
plt.ylabel('MSE')
# plt.title('Visualization of real and predicted temperature.')
# str_temp = os.path.basename(file_name).split('.')[0] + "_%05d"%i + ".png"
# path_temp = os.path.join(out_path, str_temp)
plt.show()
plt.savefig("/home/poac/lyx/Project/pic1.png", dpi=1024)
