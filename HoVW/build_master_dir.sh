MASTER='Master'
DIR_IMAGES=$MASTER'/images'
DIR_DATASET=$MASTER'/dataset'

DIR_TRAIN=$DIR_DATASET'/train'
DIR_TEST=$DIR_DATASET'/test'

DIR_OUT=$MASTER'/datasetOutput'

DIR_BOW=$DIR_OUT'/BOW'
DIR_ASS_KME=$DIR_BOW'/assignment-kme'
DIR_ASS_MSH=$DIR_BOW'/assignment-msh'
DIR_BAG_KME=$DIR_BOW'/bag-kme'
DIR_BAG_MSH=$DIR_BOW'/bag-msh'

DIR_LOGS=$DIR_OUT'/logs'
DIRC=$DIR_OUT'/descriptor'

if [ ! -d "$MASTER" ]; then
    mkdir $MASTER
    echo "mkdir "$MASTER
fi

if [ ! -d "$DIR_IMAGES" ]; then
    mkdir $DIR_IMAGES
    echo "mkdir "$DIR_IMAGES
fi

if [ ! -d "$DIR_DATASET" ]; then
    mkdir $DIR_DATASET
    echo "mkdir "$DIR_DATASET
fi

if [ ! -d "$DIR_TRAIN" ]; then
    mkdir $DIR_TRAIN 
    echo "mkdir "$DIR_TRAIN
fi

if [ ! -d "$DIR_TEST" ]; then
    mkdir $DIR_TEST
    echo "mkdir "$DIR_TEST
fi

if [ ! -d "$DIR_OUT" ]; then
    mkdir $DIR_OUT
    echo "mkdir "$DIR_OUT
fi

if [ ! -d "$DIR_BOW" ]; then
    mkdir $DIR_BOW
    echo "mkdir "$DIR_BOW
fi

if [ -d "$DIR_ASS_KME" ]; then
  rm -rf $DIR_ASS_KME
  mkdir $DIR_ASS_KME
  echo "rm -rf | mkdir "$DIR_ASS_KME
else
  mkdir $DIR_ASS_KME
  echo "mkdir "$DIR_ASS_KME
fi

if [ -d "$DIR_ASS_MSH" ]; then
  rm -rf $DIR_ASS_MSH
  mkdir $DIR_ASS_MSH
  echo "rm -rf | mkdir "$DIR_ASS_MSH
else
  mkdir $DIR_ASS_MSH
  echo "mkdir "$DIR_ASS_MSH
fi

if [ -d "$DIR_BAG_KME" ]; then
  rm -rf $DIR_BAG_KME
  mkdir $DIR_BAG_KME
  echo "rm -rf | mkdir "$DIR_BAG_KME
else
  mkdir $DIR_BAG_KME
  echo "mkdir "$DIR_BAG_KME
fi

if [ -d "$DIR_BAG_MSH" ]; then
  rm -rf $DIR_BAG_MSH
  mkdir $DIR_BAG_MSH
  echo "rm -rf | mkdir "$DIR_BAG_MSH
else
  mkdir $DIR_BAG_MSH
  echo "mkdir "$DIR_DIR_BAG_MSHIMAGES
fi

if [ -d "$DIR_LOGS" ]; then
  rm -rf $DIR_LOGS
  mkdir $DIR_LOGS
  echo "rm -rf | mkdir "$DIR_LOGS
else
  mkdir $DIR_LOGS
  echo "mkdir "$DIR_LOGS
fi

if [ -d "$DIRC" ]; then
  rm -rf $DIRC
  mkdir $DIRC
  echo "rm -rf | mkdir "$DIRC
else
  mkdir $DIRC
  echo "mkdir "$DIRC
fi

DIR_OUT=$DIR_OUT'/test'
DIR_BOW=$DIR_OUT'/BOW'
DIR_ASS_KME=$DIR_BOW'/assignment-kme'
DIR_ASS_MSH=$DIR_BOW'/assignment-msh'
DIR_BAG_KME=$DIR_BOW'/bag-kme'
DIR_BAG_MSH=$DIR_BOW'/bag-msh'

if [ ! -d "$DIR_OUT" ]; then
    mkdir $DIR_OUT
fi

if [ ! -d "$DIR_BOW" ]; then
    mkdir $DIR_BOW
fi

if [ -d "$DIR_ASS_KME" ]; then
  rm -rf $DIR_ASS_KME
  mkdir $DIR_ASS_KME
  echo "rm -rf | mkdir "$DIR_ASS_KME
else
  mkdir $DIR_ASS_KME
  echo "mkdir "$DIR_ASS_KME
fi

if [ -d "$DIR_ASS_MSH" ]; then
  rm -rf $DIR_ASS_MSH
  mkdir $DIR_ASS_MSH
  echo "rm -rf | mkdir "$DIR_ASS_MSH
else
  mkdir $DIR_ASS_MSH
  echo "mkdir "$DIR_ASS_MSH
fi

if [ -d "$DIR_BAG_KME" ]; then
  rm -rf $DIR_BAG_KME
  mkdir $DIR_BAG_KME
  echo "rm -rf | mkdir "$DIR_BAG_KME
else
  mkdir $DIR_BAG_KME
  echo "mkdir "$DIR_BAG_KME
fi

if [ -d "$DIR_BAG_MSH" ]; then
  rm -rf $DIR_BAG_MSH
  mkdir $DIR_BAG_MSH
  echo "rm -rf | mkdir "$DIR_BAG_MSH
else
  mkdir $DIR_BAG_MSH
  echo "mkdir "$DIR_BAG_MSH
fi