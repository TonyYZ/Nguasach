main.py 导入谷歌翻译输出文件

collection.csv 语言翻译结果

nguasach.xlsx 所有语言

nguasachV.csv 需要输入PSV的所有语言 给向量打词语标签

Chinese.txt 汉语词语标签 （等同于nguasachV）

nguasachVOnlyE.csv 只有espeaker支持的语言 用于自动生成读音

Thai.txt 泰语http://www.thai-language.com/?nav=dictionary&anyxlit=1 复制表格粘贴进记事本

processThai.py 从Thai.txt提取出泰语的部分

ThaiV.txt 提取结果

pronounce.py 将大部分语言的每个词语依次填充在eSpeak NG文件夹readyV.txt里 自动调用espaker生成结果在processedV.txt 最后将所有结果合并存储在resultV.csv里面，剩下梵语藏语调用相应库，日语打开词典文件，泰语导入ThaiV.txt

resultV.csv pronounce的结果

resultVT.xlsx resultV的转置矩阵（行列倒过来）放进Other

Other.csv 除了汉语的所有语言处理过后的IPA

processChn 将Chinese.txt词语全部转换为PSV (Phonetic Similarity Vectors) 可处理的格式

ChineseV.txt 在PSV文件夹里 processChn的结果

processAll 从Other.csv提取出剩下所有语言 全部转换为PSV可处理格式 名为语言名+V.txt 保存在PSV文件夹里，并交给generate.py批量为每种语言生成嵌入向量文件（词语  数1 数2 数3…… 词数之间2个空格）还可以用genTable来生成一个音节表（横轴声母 竖轴韵母）

generate.py PSV的一部分 读取一个可处理格式下的IPA文件 为每个词语生成一个对应向量（词语 数1 数2 数3…… 词数之间1个空格）得到名为语言名+Emb.txt 保存在PSV文件夹里

generateTable.py 取了generate.py的前一部分 由processAll的genTable调用 生成音节表

similarity.py PSV的一部分 读取一个嵌入向量文件 用户可以输入一个词语得到与其最接近向量对应的词语

transPhone.py 读取任意两种语言 生成对应两种Keyedvectors模型 用transvec库建立bilingual模型 将nguasachV前80%词语输入bilingual模型训练 输出剩下20%的测试结果

generateTable基于processAll生成的源语言音节表生成一个目标语言音节表（即求平均向量并转换）

thingsDict.py 生成THINGS数据集在所有语言中的对应翻译 先从MUSE多语言词典搜集 若空缺则谷歌翻译填补 再与word2word的五个备选方案做对照

thingsDictCompare.csv 当前词汇未出现再word2word选项中则列出选项 出现则标记y（yes） word2word无此词条则标记n（no）

transposeResult.py 将resultV.csv的表格转置以后保存到resultVT.csv里

sortHexagram.py 将n维向量的emb文件的每个词条按照正负转换成1或0的序列保存到语言名+hex.txt

neuralNetwork.py 负责transPhone.py的神经网络模式

compressSemantics.py 将n维向量的model.txt转换成少于n维向量的SemanticsEmb.txt

loadLarge.py 从cc.en.300.vec读取六十万个词 选取前面大约两万五千个词 去除掉废词（简写、专有名词）并加入出现在nguasach 却未出现在两万五千的词
