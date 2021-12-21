import matplotlib.pyplot as plt
import glob
import fnmatch
import os
import statistics

plt.style.use('seaborn')


folder = '20finalpredict'
data_type = '*.png'

def get_images_pre(path, extension, recursive):
    if not recursive:
        img_paths = glob.glob(path + extension)
    else:
        img_paths = []
        for root, directories, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, extension):
                img_paths.append(os.path.join(filename))

    img_paths.sort()
    return img_paths

def allocate(img_paths):
    heterozygous = []
    homozygous = []
    for i in range(len(img_paths)):
        if 'hetero' in img_paths[i]:
            heterozygous += [(img_paths[i], i)]
        else:
            homozygous += [(img_paths[i], i)]

    return heterozygous, homozygous


O_C = [(84.889, 0.08822), (2.91, 0.11977), (58.0766, 0.04410), (8.2442, 0.08729), (-14.930, 0.061), (-58.299, 0.032), (-27.844, 0.028), (-1.320, 0.159),
       (-1.296, 0.184), (-3.665, 0.119), (-81.734, 0.089), (35.598, 0.130), (2.851, 0.083), (81.881, 0.053)]

orientation = [O_C[i][0] for i in range(len(O_C))]
abs_orientation = [abs(O_C[i][0]) for i in range(len(O_C))]
coherency = [O_C[i][1] for i in range(len(O_C))]

print(orientation, coherency)

img_paths = get_images_pre(folder, extension=data_type, recursive=True)

allocate = allocate(img_paths)

print(allocate[0])
print(allocate[1])

barWidth = 0.9
bars1 = [round(orientation[allocate[0][i][1]], 3) for i in range(len(allocate[0]))]
print('bars1', bars1)
mean1 = statistics.mean(bars1)
print(mean1)
bars2 = [round(orientation[allocate[1][i][1]], 3) for i in range(len(allocate[1]))]
print('bars2',bars2)
mean2 = statistics.mean(bars2)
print(mean2)
bars = bars1 + bars2
print(bars)

allocated = [allocate[0][i][0] for i in range(len(allocate[0]))] + [allocate[1][i][0] for i in range(len(allocate[1]))]
print('allocated :',allocated)

simp_allocated = []
for i in range(len(allocated)):
    simp_allocated += [allocated[i][0:len(allocated[i])-7]]
print('simp allocated :',simp_allocated)

r1 = [i + 1 for i in range(len(bars1))]
print(r1)
r2 = [r1[-1] + i + 1 for i in range(len(bars2))]
print(r2)
r4 = r1 + r2
print(r4)

plt.bar(r1, bars1, width=barWidth, color=(0.3, 0.1, 0.4, 0.6), label='heterozygous' + ' mean :' + str(mean1))
plt.bar(r2, bars2, width=barWidth, color=(0.3, 0.5, 0.4, 0.6), label='homozygous'+ ' mean :' + str(mean2))

plt.legend()

plt.xticks([r + barWidth for r in range(len(r4))],
           simp_allocated, rotation=90)

label = [str(bars[i]) for i in range(len(r4))]

for i in range(len(r4)):
    plt.text(x=r4[i] - 0.3, y=bars[i] + 0.001, s=label[i], size=3)

plt.title('tracheas 2D Orientation in degrees')

plt.subplots_adjust(bottom=0.2, top=0.95)

plt.savefig('orientation.jpg', format='jpg', dpi=1200)
plt.show()

######################################################

barWidth = 0.9
bars1 = [round(abs_orientation[allocate[0][i][1]], 3) for i in range(len(allocate[0]))]
print(bars1)
mean1 = statistics.mean(bars1)
print(mean1)
bars2 = [round(abs_orientation[allocate[1][i][1]], 3) for i in range(len(allocate[1]))]
print(bars2)
mean2 = statistics.mean(bars2)
print(mean2)
bars = bars1 + bars2

allocated = [allocate[0][i][0] for i in range(len(allocate[0]))] + [allocate[1][i][0] for i in range(len(allocate[0]))]
print(allocated)

simp_allocated = []
for i in range(len(allocated)):
    simp_allocated += [allocated[i][0:len(allocated[i])-12]]
print(simp_allocated)

r1 = [i + 1 for i in range(len(bars1))]
print(r1)
r2 = [ r1[-1] + i + 1 for i in range(len(bars2))]
print(r2)
r4 = r1 + r2
print(r4)

plt.bar(r1, bars1, width=barWidth, color=(0.3, 0.1, 0.4, 0.6), label='heterozygous' + ' mean :' + str(mean1))
plt.bar(r2, bars2, width=barWidth, color=(0.3, 0.5, 0.4, 0.6), label='homozygous'+ ' mean :' + str(mean2))

plt.legend()

plt.xticks([r + barWidth for r in range(len(r4))],
           simp_allocated, rotation=90)

label = [str(bars[i]) for i in range(len(r4))]

for i in range(len(r4)):
    plt.text(x=r4[i] - 0.3, y=bars[i] + 0.001, s=label[i], size=3)

plt.title('tracheas 2D Orientation in degrees (abs values)')

plt.subplots_adjust(bottom=0.2, top=0.95)

plt.savefig('abs_orientation.jpg', format='jpg', dpi=1200)
plt.show()
