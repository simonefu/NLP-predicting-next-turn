from utils import *
import sys

if len(sys.argv) != 2:
	print('usage : %s <conversation> <speaker>' % sys.argv[0])
	sys.exit(1)

id_conv = sys.argv[1]

path_saving = '..' + os.path.join(  os.sep ) + 'input' + os.path.join(  os.sep )

speaker_A = create_one_side_conv(id_conv, 'A')
speaker_B = create_one_side_conv(id_conv, 'B')

frames = [speaker_A, speaker_B]

conv_AB = pd.concat(frames)
conv_AB = conv_AB.sort_values(by='start_time', ascending=True)
conv_AB = conv_AB.reset_index()

conv_AB.to_csv(path_saving + str(id_conv) + '.csv')