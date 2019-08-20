#!/usr/bin/python

from __future__ import division, print_function
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.manifold import TSNE
from multiprocessing import Pool
from six.moves import urllib
from os.path import join
from PIL import Image
from umap import UMAP
from math import ceil
from glob import glob
import tensorflow as tf
import numpy as np
import json
import os
import re
import sys
import tarfile
import psutil
import subprocess
import codecs
import argparse
from gooey import Gooey, GooeyParser
import warnings



@Gooey

def arg_define():
  parser = GooeyParser(description='Pix Plot analysis')
  parser.add_argument('image_Dir', widget='DirChooser', help='Folder containing all images prepared for analysis. Images should be in JPEG format ending in .jpg')
  parser.add_argument('model_use', widget='FileChooser', help='Select the pretrained model that you want to use.')
  parser.add_argument('clusters', action= 'store', type=int, help='Choose the number of hotspots you want pix_plot to find')
  parser.add_argument('output', widget='DirChooser', help='The folder where output files will be stored')
  parser.add_argument('method', widget='Dropdown', choices=['umap', 'tsne'], action='store', help='Select method for mapping, type umap or TSNE')
  args = parser.parse_args()
  return args 


def validate_inputs(validate_files, image_files, n_clusters):
  '''
  Make sure the inputs are valid, and warn users if they're not
  '''
  # ensure the user provided enough input images
  if len(image_files) < n_clusters:
    print('Please provide >= ' + str(n_clusters) + ' images')
    print(str(len(image_files)) + ' images were provided')
    sys.exit()

  if not validate_files:
    print(' * skipping image validation')
    return

  # test whether each input image can be processed
  print(' * validating input files')
  invalid_files = []
  for i in image_files:
    try:
      im = Image.open(i)
      response = im.size
    except Exception as exc:
      invalid_files.append(i)
  if invalid_files:
    message = '\n\nThe following files could not be processed:'
    message += '\n  ! ' + '\n  ! '.join(invalid_files) + '\n'
    message += 'Please remove these files and reprocess your images.'
    print(message)
    sys.exit()


def create_output_dirs(output_dir):
  '''
  Create each of the required output dirs
  '''
  dirs = ['image_vectors', 'atlas_files', 'thumbs']
  for i in dirs:
    ensure_dir_exists( join(output_dir, i) )
  # make subdirectories for each image thumb size
  for i in sizes:
    ensure_dir_exists( join(output_dir, 'thumbs', str(i) + 'px') )


def create_image_thumbs(image_files, output_dir, rewrite_image_thumbs, errored_images, thumb_sizes):
  '''
  Create output thumbs in 32px, 64px, and 128px
  '''
  print(' * creating image thumbs')
  resize_args = []
  for image_file in image_files:

    resize_args.append([image_file, thumb_sizes, output_dir])
  pool = Pool()
  for result in pool.imap(resize_thumb, resize_args):
    if result:
      errored_images.add( get_filename(result) )



def create_image_vectors(errored_images, image_files, output_dir):
  '''
  Create one image vector for each input file
  '''
  download_inception(model_use, model_dir)
  create_tf_graph(model_dir)

  print(' * creating image vectors')
  with tf.Session() as sess:
    for image_index, image in enumerate(image_files):
      try:
        print(' * processing image', image_index+1, 'of', len(image_files))
        outfile_name = os.path.basename(image) + '.npy'
        out_path = join(output_dir, 'image_vectors', outfile_name)
        if os.path.exists(out_path) and not rewrite_image_vectors:
          continue
        # save the penultimate inception tensor/layer of the current image
        with tf.gfile.FastGFile(image, 'rb') as f:
          data = {'DecodeJpeg/contents:0': f.read()}
          feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
          feature_vector = np.squeeze( sess.run(feature_tensor, data) )
          np.save(out_path, feature_vector)
        # close the open files
        for open_file in psutil.Process().open_files():
          file_handler = getattr(open_file, 'fd')
          os.close(file_handler)
      except Exception as exc:
        errored_images.add( get_filename(image) )
        print(' * image', get_ascii_chars(image), 'hit a snag', exc)


def download_inception(model_use, model_dir):
  '''
  Download the inception model to FLAGS.model_dir
  '''
  print(' * verifying inception model availability')
  inception_path = model_use
  dest_directory = model_dir
  ensure_dir_exists(dest_directory)
  filename = inception_path.split('/')[-1]
  filepath = join(dest_directory, filename)
  if not os.path.exists(filepath):
    def progress(count, block_size, total_size):
      percent = float(count * block_size) / float(total_size) * 100.0
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, percent))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(inception_path, filepath, progress)
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_tf_graph(model_dir):
  '''
  Create a graph from the saved graph_def.pb
  '''
  print(' * creating tf graph')
  graph_path = join(model_dir, 'classify_image_graph_def.pb')
  with tf.gfile.FastGFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def get_2d_image_positions(image_vectors):
  '''
  Create a 2d embedding of the image vectors
  '''
  print(' * calculating 2D image positions')
  model = build_model(method, image_vectors)
  return get_image_positions(model, errored_images, output_dir, v_files)


def load_image_vectors(output_dir):
  '''
  Return all image vectors
  '''
  print(' * loading image vectors')
  vector_files = glob( join(output_dir, 'image_vectors', '*') )
  image_vectors=[]
  for c, i in enumerate(vector_files):
    image_vectors.append(np.load(i))
    print(' * loaded', c+1, 'of', len(vector_files), 'image vectors')
  return vector_files, image_vectors


def build_model(method, image_vectors):
  '''
  Build a 2d projection of the `image_vectors`
  '''
  print(' * building 2D projection')
  if method == 'tsne':
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    return model.fit_transform( np.array(image_vectors) )

  elif method == 'umap':
    model = UMAP(n_neighbors=25, min_dist=0.00001, metric='correlation')
    return model.fit_transform( np.array(image_vectors) )


def get_image_positions(fit_model, errored_images, output_dir, vector_files):
  '''
  Write a JSON file that indicates the 2d position of each image
  '''
  print(' * writing JSON file')
  image_positions = []
  for c, i in enumerate(fit_model):
    img = get_filename(vector_files[c])
    if img in errored_images:
      continue
    thumb_path = os.path.join(output_dir, 'thumbs', '32px', img)
    with Image.open(thumb_path) as image:
      width, height = image.size
    # Add the image name, x offset, y offset
    image_positions.append([
      os.path.splitext(os.path.basename(img))[0],
      int(i[0] * 100),
      int(i[1] * 100),
      width,
      height
    ])
  return image_positions


def get_centroids(vector_files, n_clusters, image_vectors):
  '''
  Use KMeans clustering to find n centroid images
  that represent the center of an image cluster
  '''
  print(' * calculating ' + str(n_clusters) + ' clusters')
  model = KMeans(n_clusters)
  X = np.array(image_vectors)
  fit_model = model.fit(X)
  centroids = fit_model.cluster_centers_
  print(centroids)
  # find the points closest to the cluster centroids
  closest, _ = pairwise_distances_argmin_min(centroids, X)
  print(closest)
  centroid_paths = [vector_files[i] for i in closest]
  centroid_json = []
  for c, i in enumerate(centroid_paths):
    centroid_json.append({
      'img': get_filename(i),
      'label': 'Cluster ' + str(c+1)
    })
  return centroid_json


def write_json(output_dir, vector_files, image_vectors, n_clusters):
  '''
  Write a JSON file with image positions, the number of atlas files
  in each size, and the centroids of the k means clusters
  '''
  print(' * writing main JSON plot data file')
  out_path = join(output_dir, 'plot_data.json')
  with open(out_path, 'w') as out:
    json.dump({
      'centroids': get_centroids(vector_files, n_clusters, image_vectors),
      'positions': get_2d_image_positions(image_vectors),
      'atlas_counts': get_atlas_counts(vector_files),
    }, out)


def get_atlas_counts(vector_files):
  file_count = len(vector_files)
  return {
    '32px': ceil( file_count / (64**2) ),
    '64px': ceil( file_count / (32**2) )
  }


def create_atlas_files(sizes):
  '''
  Create image atlas files in each required size
  '''
  print(' * creating atlas files')
  atlas_group_imgs = []
  for thumb_size in sizes[1:-1]:
    # identify the images for this atlas group
    atlas_thumbs = get_atlas_thumbs(thumb_size, output_dir)
    atlas_group_imgs.append(len(atlas_thumbs))
    write_atlas_files(thumb_size, atlas_thumbs, rewrite_atlas_files, output_dir)
  # assert all image atlas files have the same number of images
  assert all(i == atlas_group_imgs[0] for i in atlas_group_imgs)


def get_atlas_thumbs(thumb_size, output_dir):
  thumbs = []
  thumb_dir = join(output_dir, 'thumbs', str(thumb_size) + 'px')
  with open(join(output_dir, 'plot_data.json')) as f:
    for i in json.load(f)['positions']:
      thumbs.append( join(thumb_dir, i[0] + '.jpg') )
  return thumbs


def get_magick_command(cmd):
  '''
  Return the specified imagemagick command prefaced with magick if
  the user is on Windows
  '''
  if os.name == 'nt':
    return 'magick ' + cmd
  return cmd


def subdivide(l, n):
  '''
  Return n-sized sublists from iterable l
  '''
  n = int(n)
  for i in range(0, len(l), n):
    yield l[i:i + n]


def get_ascii_chars(s):
  '''
  Return a string that contains the ascii characters from string `s`
  '''
  return ''.join(i for i in s if ord(i) < 128)


def get_filename(path):
  '''
  Return the root filename of `path` without file extension
  '''
  return os.path.splitext( os.path.basename(path) )[0]


def ensure_dir_exists(directory):
  '''
  Create the input directory if it doesn't exist
  '''
  if not os.path.exists(directory):
    os.makedirs(directory)


def limit_float(f):
  '''
  Limit the float point precision of float value f
  '''
  return int(f*10000)/10000


def get_files(image_Dir):
  image_names = []
  image_names = os.listdir(image_Dir)
  image_paths = []
  nonjpg = []
  extension = ['jpg', 'png', 'JPG', 'PNG', 'jpeg']
  for f in image_names:
    if f[-3:] in extension:
      image_paths.append(str(image_Dir) + '/' + str(f))
    else:
      nonjpg.append(f)
  return image_paths


def resize_thumb(args):
  image_file, sizes, output_dir = args
  for size in sizes:
    dim = size, size
    file, ext = os.path.splitext(image_file)
    path = []
    path = file.split('/')
    im = Image.open(image_file)
    im.thumbnail(dim)
    im.save(output_dir + '/thumbs/' + str(size) + 'px/' + path[-1] + '.jpg', "JPEG")

def make_contact_sheet(fnames,dim,photo,margins,padding):
    """\
    Make a contact sheet from a group of filenames:

    fnames       A list of names of the image files
    
    ncols        Number of columns in the contact sheet
    nrows        Number of rows in the contact sheet
    photow       The width of the photo thumbs in pixels
    photoh       The height of the photo thumbs in pixels

    marl         The left margin in pixels
    mart         The top margin in pixels
    marr         The right margin in pixels
    marb         The bottom margin in pixels

    padding      The padding between images in pixels

    returns a PIL image object.
    """

    # Calculate the size of the output image, based on the
    #  photo thumb sizes, margins, and padding
    ncols = dim[0]
    nrows = dim[1]
    photow = photo[0]
    photoh = photo[1]
    marl = margins[0]
    mart = margins[1]
    marr = margins[2]
    marb = margins[3]

    marw = marl+marr
    marh = mart+ marb

    padw = (ncols-1)*padding
    padh = (nrows-1)*padding
    isize = (ncols*photow+marw+padw,nrows*photoh+marh+padh)

    # Create the new image. The background doesn't have to be white
    white = (255,255,255)
    inew = Image.new('RGB',isize,white)

    count = 0
    # Insert each thumb:
    for irow in range(nrows):
        for icol in range(ncols):
            left = marl + icol*(photow+padding)
            right = left + photow
            upper = mart + irow*(photoh+padding)
            lower = upper + photoh
            try:
                # Read in an image and resize appropriately
                img = Image.open(fnames[count])
                img_bbox = img.getbbox()
                width = img_bbox[2] - img_bbox[0]
                height = img_bbox[3] - img_bbox[1]

                # calculate a scaling factor depending on fitting the larger dimension into the thumbnail                
                ratio = max(height/float(photoh), width/float(photow))

                newWidth = int(width/ratio)
                newHeight = int(height/ratio)
                newSize = (newWidth, newHeight)

                img = img.resize(newSize)
            except:
                break

            new_left = left
            new_upper = upper

            if ( newWidth < photow):
                new_left = int( left + ((photow - newWidth)/2))

            if ( newHeight < photoh):
                new_upper = int(upper + ((photoh - newHeight)/2))

            inew.paste(img, (new_left, new_upper))
            count += 1
    return inew

def write_atlas_files(thumb_size, image_thumbs, rewrite_atlas_files, output_dir):

    '''
    Given a thumb_size (int) and image_thumbs [file_path],
    write the total number of required atlas files at this size
    '''
    if not rewrite_atlas_files:
      return

    # build a directory for the atlas files
    out_dir = join(output_dir, 'atlas_files', str(thumb_size) + 'px')
    ensure_dir_exists(out_dir)

    # specify number of columns in a 2048 x 2048px texture
    atlas_cols = int(2048/thumb_size)

    # subdivide the image thumbs into groups
    atlas_image_groups = subdivide(image_thumbs, atlas_cols**2)

    # generate a directory for images at this size if it doesn't exist
    for idx, atlas_images in enumerate(atlas_image_groups):
      print(' * creating atlas', idx + 1, 'at size', thumb_size)
      out_path = join(out_dir, 'atlas-' + str(idx) + '.jpg')
      # write a file containing a list of images for the current montage
      tmp_file_path = join(output_dir, 'images_to_montage.txt')
      with codecs.open(tmp_file_path, 'w', encoding='utf-8') as out:
        # python 2
        try:
          out.write('\n'.join(map('"{0}"'.decode('utf-8').format, atlas_images)))
        # python 3
        except AttributeError:
          out.write('\n'.join(map('"{0}"'.format, atlas_images)))
  
      ncols,nrows = atlas_cols,atlas_cols
      dim = [ncols,nrows]
      files = get_thumbs(tmp_file_path)
      # Don't bother reading in files we aren't going to use
      if len(files) > ncols*nrows: files = files[:ncols*nrows]
      # These are all in terms of pixels:
      photow,photoh = thumb_size,thumb_size
      photo = [photow,photoh]
      margins = [5,5,5,5]
      padding = 1
      inew = make_contact_sheet(files,dim,photo,margins,padding)
      inew.save(out_path)
    try:
      os.remove(tmp_file_path)
    except Exception:
      pass


def get_thumbs(tmp_file_path):
  image_names = []
  f = open(tmp_file_path, "r")
  for line in f:
    image_names.append(line)
  image_paths = []
  nonjpg = []
  extension = ['jpg', 'png']
  for f in image_names:
    if f[-3:] in extension:
      image_paths.append(str(tmp_file_path) + '/' + str(f))
    else:
      nonjpg.append(f)
  return image_files

def get_model_dir(model_use):
  model_dir = ''
  path_list= []
  s = '/'
  path_list = model_use.split('/')
  model_dir = s.join(path_list[:-1])  
  return(model_dir)

if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  os.system('git clone https://github.com/YaleDHLab/pix-plot && cd pix-plot')
  args = arg_define()
  image_Dir = args.image_Dir
  image_files = get_files(image_Dir)
  output_dir = args.output
  model_use = args.model_use
  model_dir = get_model_dir(model_use)
  sizes = [16, 32, 64, 128]
  n_clusters = args.clusters
  errored_images = set()
  vector_files = []
  method = args.method
  rewrite_image_thumbs = False
  rewrite_image_vectors = False
  rewrite_atlas_files = True
  validate_files = True
  print(' * writing PixPlot outputs with ' + str(n_clusters) +
      ' clusters for ' + str(len(image_files)) +
      ' images to folder ' + output_dir)

  validate_inputs(validate_files, image_files, n_clusters)
  create_output_dirs(output_dir)
  create_image_thumbs(image_files, output_dir, rewrite_image_thumbs, errored_images, sizes)
  create_image_vectors(errored_images, image_files, output_dir)
  v_files, image_v = load_image_vectors(output_dir)
  print(image_v)
  write_json(output_dir, v_files, image_v, n_clusters)
  create_atlas_files(sizes)
  print('Done, launch webserver by typing "localhost:8000" into your favorite browser!')
  os.system('cd pix-plot && python -m http.server 8000')

