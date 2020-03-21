# 说明

## 标签说明

采用百度的分词、词性、ner标签体系
```
词性	含义	词性	含义	词性	含义	词性	含义
n	普通名词	f	方位名词	s	处所名词	t	时间名词
nr	人名	ns	地名	nt	机构团体名	nw	作品名
nz	其他专名	v	普通动词	vd	动副词	vn	名动词
a	形容词	ad	副形词	an	名形词	d	副词
m	数量词	q	量词	r	代词	p	介词
c	连词	u	助词	xc	其他虚词	w	标点符号

缩略词	含义	缩略词	含义	缩略词	含义	缩略词	含义
PER	人名	LOC	地名	ORG	机构名	TIME	时间
```

轻量模型，存在 少量 分词粒度/边界/语义 问题

输出结果如下：

cost time:3.0593178272247314
{'word': '1996年', 'tag': 'TIME', 'desc': '时间'}
{'word': '，', 'tag': 'w', 'desc': '标点符号'}
{'word': '曾经', 'tag': 'd', 'desc': '副词'}
{'word': '是', 'tag': 'v', 'desc': '普通动词'}
{'word': '微软', 'tag': 'ORG', 'desc': '机构名'}
{'word': '员工', 'tag': 'n', 'desc': '普通名词'}
{'word': '的', 'tag': 'u', 'desc': '助词'}
{'word': '加布·纽维尔', 'tag': 'PER', 'desc': '人名'}
{'word': '和', 'tag': 'c', 'desc': '连词'}
{'word': '麦克·哈灵顿', 'tag': 'PER', 'desc': '人名'}
{'word': '一同', 'tag': 'd', 'desc': '副词'}
{'word': '创建', 'tag': 'v', 'desc': '普通动词'}
{'word': '了', 'tag': 'u', 'desc': '助词'}
{'word': 'Valve软件公司', 'tag': 'ORG', 'desc': '机构名'}
{'word': '。', 'tag': 'w', 'desc': '标点符号'}
{'word': '他们', 'tag': 'r', 'desc': '代词'}
{'word': '在', 'tag': 'p', 'desc': '介词'}
{'word': '1996年下半年', 'tag': 'TIME', 'desc': '时间'}
{'word': '从', 'tag': 'p', 'desc': '介词'}
{'word': 'idsoftware', 'tag': 'nz', 'desc': '其他专名'}
{'word': '取得', 'tag': 'v', 'desc': '普通动词'}
{'word': '了', 'tag': 'u', 'desc': '助词'}
{'word': '雷神', 'tag': 'n', 'desc': '普通名词'}
{'word': '之', 'tag': 'u', 'desc': '助词'}
{'word': '锤', 'tag': 'n', 'desc': '普通名词'}
{'word': '引擎', 'tag': 'n', 'desc': '普通名词'}
{'word': '的', 'tag': 'u', 'desc': '助词'}
{'word': '使用', 'tag': 'vn', 'desc': '名动词'}
{'word': '许可', 'tag': 'vn', 'desc': '名动词'}
{'word': '，', 'tag': 'w', 'desc': '标点符号'}
{'word': '用来', 'tag': 'v', 'desc': '普通动词'}
{'word': '开发', 'tag': 'v', 'desc': '普通动词'}
{'word': '半条', 'tag': 'm', 'desc': '数量词'}
{'word': '命', 'tag': 'n', 'desc': '普通名词'}
{'word': '系列', 'tag': 'n', 'desc': '普通名词'}
{'word': '。', 'tag': 'w', 'desc': '标点符号'}