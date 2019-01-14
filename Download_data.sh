conda activate tfpy3
pip install --user kaggle-cli
export PATH=~/.local/bin:$PATH

cd ~
mkdir .kaggle
cd ~/.kaggle
echo "{\"username\":\"archonsh\",\"key\":\"9969e262455eda17e17564ab5c5efa6a\"}" > kaggle.json
chmod 600 ~/.kaggle/kaggle.json

cd ~
kaggle datasets download -d utmhikari/doubanmovieshortcomments
unzip doubanmovieshortcomments.zip
rm doubanmovieshortcomments.zip