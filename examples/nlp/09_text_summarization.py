#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文文本摘要实战 - GPT 生成项目
=================================
项目名称：新闻标题生成与文章摘要
任务：使用 GPT-2 进行文本生成，实现新闻标题生成和文章摘要

学习目标：
- 理解文本生成任务与分类/标注的区别
- 掌握自回归语言模型的训练方式
- 学习不同的生成解码策略（Greedy、Beam Search、Sampling）
- 了解文本生成的评估指标（ROUGE）

文本生成 vs 分类/标注：
- 分类：输入 → 输出1个标签（判断）
- 标注：输入 → 输出N个标签（识别）
- 生成：输入 → 输出序列（创作）
"""

import os

# ========== 配置 Hugging Face 镜像源 ==========
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', '..', '.cache', 'huggingface')
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

print(f"🌐 使用镜像源: {os.environ['HF_ENDPOINT']}")
print(f"📁 模型缓存目录: {os.environ['HF_HOME']}")

# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm

print("=" * 70)
print("中文文本摘要实战 - GPT 文本生成")
print("=" * 70)

# ============================================================
# 第一部分：数据准备 - 文本对格式
# ============================================================
print("\n" + "=" * 70)
print("第一部分：数据准备 - 文章-摘要对")
print("=" * 70)

print("""
【文本生成数据格式】

与分类/标注不同，生成任务需要"输入-输出"文本对：

格式： article [SEP] summary [EOS]
示例： 本文介绍了人工智能的发展历程... [SEP] AI发展历程回顾 [EOS]

模型学习的是：给定文章，生成摘要的条件概率
P(summary | article) = P(摘要第一个字|文章) × P(摘要第二个字|文章+第一个字) × ...

这种自回归方式，模型逐个生成每个字。
""")

# 模拟新闻摘要数据
raw_data = [
    # 科技新闻
    {
        "article": "阿里巴巴集团今日宣布，将在未来三年内投入1000亿元人民币用于云计算基础设施建设。该计划旨在提升阿里云在全球市场的竞争力，并支持更多企业实现数字化转型。阿里云智能事业群总裁张建锋表示，此次投资将重点用于数据中心建设、芯片研发和人才培养。",
        "summary": "阿里云计划三年投资千亿建设云基础设施"
    },
    {
        "article": "苹果公司于今日凌晨举行秋季发布会，正式发布了iPhone 15系列手机。新款iPhone采用了全新的USB-C接口，取代了使用多年的Lightning接口。此外，iPhone 15 Pro系列搭载了全新的A17 Pro芯片，性能提升显著。",
        "summary": "iPhone 15发布：采用USB-C接口和A17 Pro芯片"
    },
    {
        "article": "特斯拉CEO埃隆·马斯克在社交媒体平台X上宣布，特斯拉全自动驾驶（FSD）Beta版将在下个月向北美所有付费用户开放。马斯克表示，FSD目前的测试里程已超过5亿英里，安全性和可靠性得到了大幅提升。",
        "summary": "特斯拉FSD Beta下月向北美用户全面开放"
    },
    {
        "article": "OpenAI今日宣布推出GPT-4 Turbo模型，该模型相比GPT-4具有更强的推理能力和更低的成本。新模型支持128K上下文窗口，相当于可以一次性处理300页文档。同时，OpenAI还推出了GPTs功能，允许用户创建自定义的GPT应用。",
        "summary": "OpenAI发布GPT-4 Turbo：更强更便宜"
    },
    {
        "article": "微软公司宣布完成对动视暴雪的收购交易，这笔交易价值687亿美元，是游戏行业历史上最大的一笔收购。微软表示，将把动视暴雪的游戏加入Xbox Game Pass订阅服务，为玩家提供更多优质游戏内容。",
        "summary": "微软687亿美元收购动视暴雪完成"
    },
    
    # 财经新闻
    {
        "article": "中国人民银行宣布，决定于2024年2月5日下调金融机构存款准备金率0.5个百分点。此次降准将向市场释放长期资金约1万亿元，有助于降低实体经济融资成本，支持经济稳定增长。",
        "summary": "央行降准0.5个百分点释放万亿资金"
    },
    {
        "article": "美股三大指数今日集体收涨，道琼斯指数上涨1.2%，纳斯达克指数上涨1.8%，标普500指数上涨1.5%。科技股表现强劲，英伟达股价创历史新高，市值突破1.8万亿美元。市场分析师认为，通胀数据好于预期是推动股市上涨的主要原因。",
        "summary": "美股三大指数收涨科技股领涨"
    },
    {
        "article": "国家统计局今日发布数据显示，2023年全国规模以上工业企业实现利润总额76858.3亿元，比上年下降2.3%，降幅比1-11月份收窄2.1个百分点。12月份，规模以上工业企业利润同比增长16.8%，连续5个月实现正增长。",
        "summary": "2023年工业企业利润下降2.3%降幅收窄"
    },
    {
        "article": "新能源汽车品牌比亚迪今日发布2023年销量数据，全年销售新能源汽车302.4万辆，同比增长61.9%，超越特斯拉成为全球新能源汽车销量冠军。其中，12月单月销量达到34.1万辆，创历史新高。",
        "summary": "比亚迪2023年销量超302万辆成全球新能源冠军"
    },
    {
        "article": "国际油价今日大幅上涨，布伦特原油期货价格上涨3.5%，报收于每桶85.5美元。WTI原油期货价格上涨3.2%，报收于每桶80.2美元。分析师表示，红海地区地缘政治风险加剧是推高油价的主要原因。",
        "summary": "国际油价大涨超3%因地缘政治风险"
    },
    
    # 社会新闻
    {
        "article": "教育部今日发布通知，要求各地中小学严格落实课间休息制度，不得随意缩短课间活动时间。通知强调，要保证学生每天校内体育活动不少于1小时，促进学生身心健康发展。",
        "summary": "教育部要求保障中小学生课间活动时间"
    },
    {
        "article": "国家卫健委发布最新数据显示，我国居民人均预期寿命达到78.2岁，比2015年提高了1.8岁。其中，上海、北京、天津等地人均预期寿命超过80岁，达到发达国家水平。",
        "summary": "我国人均预期寿命达78.2岁"
    },
    {
        "article": "交通运输部数据显示，2024年春运40天，全社会跨区域人员流动量预计达到90亿人次，创历史新高。其中，铁路客运量预计4.8亿人次，公路出行量预计71亿人次，水路和民航客运量预计也有显著增长。",
        "summary": "2024春运人员流动量预计达90亿人次创新高"
    },
    {
        "article": "据国家电影局统计，2024年春节档电影票房达到80.16亿元，观影人次1.63亿，刷新了中国影史春节档票房纪录。《热辣滚烫》《飞驰人生2》《熊出没·逆转时空》位列票房前三名。",
        "summary": "2024春节档票房破80亿创历史新高"
    },
    {
        "article": "中国航天科技集团宣布，嫦娥六号探测器将于今年上半年发射，执行月球背面采样返回任务。这将是人类首次从月球背面采集样本返回地球，对研究月球形成和演化具有重要意义。",
        "summary": "嫦娥六号今年上半年发射将采月球背面样本"
    },
    
    # 体育新闻
    {
        "article": "中国男足在亚洲杯小组赛中以0比1不敌卡塔尔队，三场比赛仅取得2平1负的战绩，积2分排名小组第三。由于净胜球劣势，国足基本无缘以成绩最好的小组第三名身份晋级淘汰赛，创造了参加亚洲杯以来的最差战绩。",
        "summary": "国足亚洲杯出局创历史最差战绩"
    },
    {
        "article": "在澳大利亚网球公开赛男单决赛中，意大利选手辛纳以3比2逆转战胜俄罗斯选手梅德韦杰夫，夺得个人首个大满贯冠军。辛纳成为首位夺得澳网男单冠军的意大利人，也是公开赛年代最年轻的澳网冠军之一。",
        "summary": "辛纳首夺澳网冠军成意大利第一人"
    },
    {
        "article": "NBA常规赛继续进行，湖人队主场以136比105大胜老鹰队。詹姆斯砍下25分7篮板10助攻的准三双数据，浓眉哥戴维斯贡献22分15篮板。此役过后，湖人队战绩提升至24胜23负，排名西部第九。",
        "summary": "湖人31分大胜老鹰詹姆斯25+7+10"
    },
    {
        "article": "据多家媒体报道，梅西将在今年2月率迈阿密国际队访问香港，参加一场友谊赛。这是梅西继去年加盟美职联后首次访问亚洲，预计将吸引大量球迷到场观战。香港文旅局表示，正在与主办方协商相关事宜。",
        "summary": "梅西2月将访问香港参加友谊赛"
    },
    {
        "article": "国际奥委会宣布，2024年巴黎奥运会将首次引入电子竞技项目作为表演项目，包括《英雄联盟》《王者荣耀》等游戏。国际奥委会主席巴赫表示，这标志着奥委会对电竞的认可，未来可能将电竞纳入正式比赛项目。",
        "summary": "巴黎奥运会首次引入电竞表演项目"
    },
    
    # 文化娱乐
    {
        "article": "据国家图书馆消息，馆藏古籍数字化项目已完成10万页珍贵古籍的数字化工作，其中包括《永乐大典》副本、《四库全书》底本等珍贵文献。这些数字化资源将免费向公众开放，读者可通过国家图书馆网站在线阅读。",
        "summary": "国家图书馆10万页古籍数字化完成免费开放"
    },
    {
        "article": "第96届奥斯卡金像奖提名名单今日揭晓，电影《奥本海默》以13项提名领跑，包括最佳影片、最佳导演、最佳男主角等重要奖项。《可怜的东西》获得11项提名紧随其后。颁奖典礼将于3月10日举行。",
        "summary": "《奥本海默》领跑奥斯卡提名获13项提名"
    },
    {
        "article": "著名歌手周杰伦宣布将举办世界巡回演唱会，首站定于今年4月在杭州举行。据悉，此次演唱会将采用全新的舞台设计和视觉效果，为歌迷带来全新的音乐体验。门票预售将于下月开启，预计将一票难求。",
        "summary": "周杰伦世界巡演首站4月杭州开唱"
    },
    {
        "article": "国产动画电影《哪吒之魔童降世》的续集《哪吒2》宣布定档今年暑期档上映。前作曾在2019年创下50亿元票房纪录，成为中国影史票房最高的动画电影。导演饺子表示，续集将在特效和故事上都有全面升级。",
        "summary": "《哪吒2》定档暑期前作票房纪录待刷新"
    },
    {
        "article": "故宫博物院发布公告，宣布将闭馆时间延长至每天17:30，并增加夜游场次。新措施将于下月1日起实施，旨在满足游客参观需求，同时更好地保护文物安全。游客需通过网上预约购票，现场不售票。",
        "summary": "故宫延长开放时间下月起实施夜游"
    },
    
    # 扩展：更多科技新闻
    {
        "article": "华为今日正式发布鸿蒙操作系统星河版，标志着鸿蒙生态迈入全新阶段。余承东表示，目前已有超过200家应用厂商启动鸿蒙原生应用开发，覆盖社交、金融、政务等多个领域。",
        "summary": "华为发布鸿蒙星河版超200家厂商启动原生应用开发"
    },
    {
        "article": "小米汽车SU7正式发布，起售价21.59万元。雷军表示，小米SU7定位C级高性能生态科技轿车，零百加速仅需2.78秒，CLTC续航里程可达800公里。首日预订量突破8万台。",
        "summary": "小米SU7正式发布起售价21.59万首日预订破8万台"
    },
    {
        "article": "字节跳动旗下AI大模型豆包宣布升级，新版本在中文理解和生成能力上大幅提升。豆包已接入字节跳动多款产品，日均处理文本量超过1200亿字，成为国内使用量最大的AI助手之一。",
        "summary": "字节豆包大模型升级日均处理文本超1200亿字"
    },
    {
        "article": "大疆创新发布新款航拍无人机Mavic 4 Pro，搭载三摄系统，支持8K视频录制。该机型配备全新图传系统，最远传输距离达30公里，续航时间提升至48分钟，售价13888元起。",
        "summary": "大疆发布Mavic 4 Pro无人机支持8K录制售价13888起"
    },
    {
        "article": "谷歌DeepMind发布新一代AlphaFold 3模型，能够预测蛋白质、DNA、RNA等生物分子的结构和相互作用。该模型有望加速新药研发进程，帮助科学家更好地理解生命机制。",
        "summary": "谷歌AlphaFold 3发布可预测生物分子结构和相互作用"
    },
    
    # 扩展：更多财经新闻
    {
        "article": "证监会发布《关于加强上市公司监管的意见》，明确提出严厉打击财务造假、严格规范大股东减持行为、推动上市公司分红回购等措施。新政旨在提升上市公司质量，保护中小投资者权益。",
        "summary": "证监会发文严打财务造假规范大股东减持"
    },
    {
        "article": "美联储宣布将联邦基金利率目标区间维持在5.25%至5.5%不变，符合市场预期。这是美联储连续第四次维持利率不变。鲍威尔表示，在确信通胀率持续向2%迈进之前，降息并不合适。",
        "summary": "美联储连续第四次维持利率不变暗示暂不考虑降息"
    },
    {
        "article": " Goldman Sachs发布研究报告，上调中国经济增长预期，预计2024年中国GDP增速将达4.8%，高于此前预测的4.5%。报告认为，消费复苏和制造业升级将成为主要增长动力。",
        "summary": "高盛上调中国经济增长预期至4.8%"
    },
    {
        "article": "台积电宣布将在日本熊本建设第二座晶圆厂，投资规模预计超过2万亿日元。新工厂将采用6纳米制程技术，预计2027年投产。这将是台积电在日本的最大单笔投资。",
        "summary": "台积电宣布在日建第二座晶圆厂投资超2万亿日元"
    },
    {
        "article": "跨境电商巨头SHEIN宣布将在巴西投资7.5亿雷亚尔，建设本地化供应链。该投资预计将为巴西创造超过1万个就业岗位。SHEIN表示，这是其全球化战略的重要一步。",
        "summary": "SHEIN宣布在巴西投资7.5亿雷亚尔建设本地化供应链"
    },
    
    # 扩展：更多社会新闻
    {
        "article": "全国多地出现极端高温天气，中央气象台连续发布高温红色预警。北京、上海、广州等城市最高气温突破40摄氏度。气象专家提醒公众减少户外活动，注意防暑降温，警惕热射病。",
        "summary": "全国多地高温突破40度中央气象台发布红色预警"
    },
    {
        "article": "民政部发布《婚姻登记条例》修订草案，拟取消户口簿作为婚姻登记的必要材料。新规实施后，内地居民结婚登记只需携带身份证即可办理，简化登记流程。",
        "summary": "结婚登记拟取消户口簿要求仅需身份证即可办理"
    },
    {
        "article": "人社部发布2024年第一季度全国招聘大于求职的100个职业排行，营销员、快递员、保洁员位列前三。人工智能训练师、智能网联汽车测试员等新职业首次进入排行。",
        "summary": "一季度最缺工职业排行发布人工智能训练师首次入榜"
    },
    {
        "article": "国家医保局通报2023年医保基金飞行检查情况，共查处违法违规定点医药机构3297家，追回医保基金超30亿元。欺诈骗保、过度诊疗等问题仍较为突出。",
        "summary": "2023年医保飞行检查追回基金超30亿查处机构3297家"
    },
    {
        "article": "我国首条跨海高铁福厦高铁正式开通运营，福州至厦门最快55分钟可达。该高铁全长277公里，设计时速350公里，是我国首条设计时速350公里的跨海高铁。",
        "summary": "福厦高铁正式开通福州至厦门最快55分钟可达"
    },
    
    # 扩展：更多体育新闻
    {
        "article": "曼城在欧冠决赛中以1比0战胜国际米兰，时隔12年再度捧起欧冠奖杯。罗德里打入全场唯一进球，德布劳内因伤提前离场。曼城本赛季实现英超、足总杯、欧冠三冠王伟业。",
        "summary": "曼城1比0国米夺得欧冠本赛季成就三冠王"
    },
    {
        "article": "德约科维奇在法网男单决赛中3比0横扫鲁德，夺得个人第23座大满贯冠军，超越纳达尔独居男子网坛历史第一。这也是德约科维奇第三次完成全满贯壮举。",
        "summary": "德约法网夺冠豪取第23座大满贯超越纳达尔独居历史第一"
    },
    {
        "article": "中国女足在世界杯小组赛最后一轮中以1比6惨败给英格兰队，三战一胜两负排名小组第三，无缘晋级16强。王霜打入中国队唯一进球，赛后泪洒赛场。",
        "summary": "中国女足1比6英格兰无缘16强王霜进球难救主"
    },
    {
        "article": "F1新加坡大奖赛结束，塞恩斯夺得冠军，诺里斯、汉密尔顿分列二三位。中国车手周冠宇以第12名完赛。本站过后，维斯塔潘提前六站锁定年度车手总冠军。",
        "summary": "F1新加坡站塞恩斯夺冠维斯塔潘提前锁定年度冠军"
    },
    {
        "article": "世界游泳锦标赛落幕，中国队以20金8银12铜的成绩位列金牌榜第一，创造历史最佳战绩。覃海洋独揽5金，张雨霏、汪顺等选手也有多金入账。",
        "summary": "游泳世锦赛落幕中国队20金位列金牌榜第一创历史最佳"
    },
    
    # 扩展：更多文化娱乐新闻
    {
        "article": "作家刘慈欣的科幻小说《三体》改编剧集在Netflix正式上线，引发全球观看热潮。该剧制作成本高达1.6亿美元，是全球制作成本最高的剧集之一。国内外观众评价呈现两极分化。",
        "summary": "Netflix版《三体》上线制作成本1.6亿美元评价两极分化"
    },
    {
        "article": "歌手刀郎发布新专辑《山歌寥哉》，其中《罗刹海市》一曲火爆全网。歌曲取材自《聊斋志异》，歌词讽刺意味浓厚，被网友解读为影射娱乐圈乱象。播放量已突破50亿次。",
        "summary": "刀郎新歌《罗刹海市》爆红播放量突破50亿次"
    },
    {
        "article": "王家卫执导的电视剧《繁花》正式收官，豆瓣评分高达8.7分。该剧改编自金宇澄同名小说，以上海为背景，讲述上世纪九十年代的故事。胡歌、马伊琍、唐嫣等主演表现出色。",
        "summary": "《繁花》收官豆瓣评分8.7分胡歌马伊琍主演获好评"
    },
    {
        "article": "著名主持人董卿宣布暂别央视舞台，引发网友热议。董卿主持过《中国诗词大会》《朗读者》等多档口碑节目，陪伴观众十余年。她表示将回归家庭，陪伴家人。",
        "summary": "董卿宣布暂别央视曾主持《中国诗词大会》等节目"
    },
    {
        "article": "B站举办2023年度百大UP主颁奖典礼，罗翔、何同学、影视飓风等创作者入选。今年百大评选标准更注重内容质量和社区贡献，知识类、科普类UP主占比显著提升。",
        "summary": "B站2023百大UP主揭晓罗翔何同学等入选知识类占比提升"
    },
    
    # 扩展：国际新闻
    {
        "article": "联合国安理会通过加沙停火决议，要求各方立即停止敌对行动。中国、俄罗斯等13国赞成，美国弃权。以色列方面表示拒绝接受该决议，将继续军事行动。",
        "summary": "联合国安理会通过加沙停火决议美国弃权以色列拒绝接受"
    },
    {
        "article": "印度正式超越英国成为全球第五大经济体，GDP总量超过3.7万亿美元。IMF预测印度将在2027年超过日本成为全球第三大经济体。印度股市市值也创历史新高。",
        "summary": "印度正式超越英国成为全球第五大经济体"
    },
    {
        "article": "日本核污染水第三轮排海结束，累计排放量超过2.3万吨。中国、韩国等周边国家对此表示强烈反对。日方声称排放符合国际标准，但多项检测数据显示放射性物质超标。",
        "summary": "日本核污染水第三轮排海结束累计排放超2.3万吨"
    },
    {
        "article": "德国宣布移除华为5G设备，要求运营商在2026年前完成更换。此举可能给德国电信运营商带来超过50亿欧元损失。中方对此表示严重关切，称这是歧视性做法。",
        "summary": "德国宣布移除华为5G设备2026年前完成更换损失或超50亿欧"
    },
    {
        "article": "瑞典正式加入北约，成为该组织第32个成员国。这是北约自2020年北马其顿加入以来首次扩员。俄罗斯表示，将采取一切必要措施保障自身安全。",
        "summary": "瑞典正式加入北约成为第32个成员国俄罗斯表态将采取措施"
    },
]

print(f"\n📊 数据集统计:")
print(f"   总样本数: {len(raw_data)}")

# 统计文章和摘要长度
article_lengths = [len(item["article"]) for item in raw_data]
summary_lengths = [len(item["summary"]) for item in raw_data]

print(f"   文章平均长度: {np.mean(article_lengths):.0f} 字符")
print(f"   摘要平均长度: {np.mean(summary_lengths):.0f} 字符")
print(f"   压缩率: {np.mean(summary_lengths)/np.mean(article_lengths)*100:.1f}%")

# 划分数据集
train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)

print(f"\n📈 数据划分:")
print(f"   训练集: {len(train_data)} 条")
print(f"   测试集: {len(test_data)} 条")

# 显示一个样本
print(f"\n[SAMPLE] 样本示例:")
print(f"   文章: {train_data[0]['article'][:50]}...")
print(f"   摘要: {train_data[0]['summary']}")

# ============================================================
# 第二部分：创建 Dataset 和 DataLoader
# ============================================================
print("\n" + "=" * 70)
print("第二部分：创建 Dataset 和 DataLoader")
print("=" * 70)

print("""
【文本生成的数据处理】

对于GPT等自回归模型，我们将输入和输出拼接：
输入格式: [文章] [SEP] [摘要] [EOS]

模型学习的是：给定[文章][SEP]，预测接下来的每个字
- 文章部分：只计算loss但不生成（已有内容）
- 摘要部分：逐字生成，每个位置预测下一个token

特殊token：
- [SEP]: 分隔文章和摘要
- [EOS]: 结束标记（End of Sequence）
""")


class SummarizationDataset(Dataset):
    """文本摘要数据集"""
    
    def __init__(self, data, tokenizer, max_article_len=200, max_summary_len=50):
        self.data = data
        self.tokenizer = tokenizer
        self.max_article_len = max_article_len
        self.max_summary_len = max_summary_len
        
        # GPT2没有专门的[SEP]和[EOS]，我们用特殊token代替
        # 或者用换行符等自然分隔
        self.sep_token = "\n摘要："  # 用中文作为分隔符
        self.eos_token = tokenizer.eos_token  # <|endoftext|>
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        article = item["article"]
        summary = item["summary"]
        
        # 构建完整序列: [文章]\n摘要：[摘要]<|endoftext|>
        full_text = article + self.sep_token + summary + self.eos_token
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_article_len + self.max_summary_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        # 构建labels：和input_ids相同，但文章部分设为-100（不计算loss）
        # 找到分隔符位置
        sep_encoding = self.tokenizer(
            article + self.sep_token,
            add_special_tokens=False,
            return_tensors='pt'
        )
        article_len = len(sep_encoding['input_ids'][0])
        
        labels = input_ids.clone()
        # 文章部分（包括分隔符）不计算loss
        labels[:article_len] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'article_len': article_len  # 记录文章长度，用于生成时截断
        }


print("\n[TOKENIZER] 加载 GPT2 Tokenizer...")
# 使用中文GPT2或标准GPT2
try:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print("   使用标准 GPT2 Tokenizer")
except:
    # 如果下载失败，使用bert的tokenizer（效果会差一些）
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    print("   [警告] 使用 BERT Tokenizer 作为备选（效果可能不佳）")

# GPT2没有pad_token，设置为eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"   设置 pad_token = eos_token ({tokenizer.eos_token})")

print(f"   词表大小: {len(tokenizer)}")
print(f"   EOS token: {tokenizer.eos_token}")

# 创建 Dataset
train_dataset = SummarizationDataset(train_data, tokenizer)
test_dataset = SummarizationDataset(test_data, tokenizer)

print(f"\n📦 Dataset 创建完成:")
print(f"   训练样本: {len(train_dataset)}")
print(f"   测试样本: {len(test_dataset)}")

# 查看一个样本
sample = train_dataset[0]
print(f"\n[SAMPLE] 样本示例:")
print(f"   Input IDs 形状: {sample['input_ids'].shape}")
print(f"   Labels 形状: {sample['labels'].shape}")
print(f"   文章长度（tokens）: {sample['article_len']}")

# 解码查看
decoded = tokenizer.decode(sample['input_ids'])
print(f"   解码文本: {decoded[:80]}...")

# 先定义默认batch_size，后续根据设备调整
default_batch_size = 4 if torch.cuda.is_available() else 2

train_loader = DataLoader(
    train_dataset,
    batch_size=default_batch_size,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=default_batch_size,
    shuffle=False,
    num_workers=0
)

print(f"\n📦 DataLoader 创建完成:")
print(f"   初始批次大小: {default_batch_size}")
print(f"   训练批次: {len(train_loader)}")
print(f"   测试批次: {len(test_loader)}")

# ============================================================
# 第三部分：加载 GPT2 模型
# ============================================================
print("\n" + "=" * 70)
print("第三部分：加载 GPT2 生成模型")
print("=" * 70)

print("""
【GPT2 文本生成原理】

GPT2 是 Decoder-only 的自回归语言模型：

结构:
Input:  [CLS] 今天 天气 真好 [SEP] 适合 [MASK]
        ↓ Embedding
        ↓ Transformer Decoder × 12层
        ↓ Language Model Head
Output: logits[今天] logits[天气] logits[真好] logits[适合] logits[出门]
        ↑
        每个位置预测下一个token的概率

训练目标:
最大化: P(出门 | 今天天气真好适合) 
      × P(适合 | 今天天气真好)
      × P(真好 | 今天天气)
      × ...

生成时:
从[文章][SEP]开始，逐个采样生成每个字，直到[EOS]
""")

print("\n🤖 加载 GPT2...")
try:
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    print("   成功加载标准 GPT2")
except:
    print("   [警告] GPT2加载失败，使用小型模型...")
    from transformers import GPT2Config
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=512,
        n_ctx=512,
        n_embd=256,
        n_layer=6,
        n_head=8
    )
    model = GPT2LMHeadModel(config)
    print("   使用随机初始化的GPT2-small")

print(f"   模型类型: {type(model).__name__}")
print(f"   参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ========== CUDA 优化设置 ==========
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n💻 使用设备: {device}")

if torch.cuda.is_available():
    print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"   GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   CUDA版本: {torch.version.cuda}")
    # 启用cuDNN自动优化
    torch.backends.cudnn.benchmark = True
    print(f"   cuDNN自动优化: 已启用")
else:
    print(f"   [警告] 未检测到CUDA，将使用CPU训练（较慢）")

model = model.to(device)

# ============================================================
# 第四部分：训练配置
# ============================================================
print("\n" + "=" * 70)
print("第四部分：训练配置")
print("=" * 70)

# ========== 训练参数优化 ==========
# 数据集40条，batch_size=4，每轮10步，30轮=300步，有利于模型收敛
EPOCHS = 30 if torch.cuda.is_available() else 10  # GPU训练更多轮数，CPU适当减少
LEARNING_RATE = 2e-5  # 降低学习率，更稳定的训练
WARMUP_STEPS = 50    # 预热步数
MAX_GRAD_NORM = 1.0

# 根据设备调整batch_size
if torch.cuda.is_available():
    BATCH_SIZE = 4  # GPU可用更大的batch_size
    print(f"   GPU训练模式: batch_size={BATCH_SIZE}, epochs={EPOCHS}")
else:
    BATCH_SIZE = 2  # CPU保持小batch_size
    print(f"   CPU训练模式: batch_size={BATCH_SIZE}, epochs={EPOCHS}")

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps
)

print(f"\n⚙️ 训练配置:")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   总训练步数: {total_steps}")
print(f"   Warmup步数: {WARMUP_STEPS}")

print("""
【文本生成的loss计算】

与分类任务不同：
- 分类: CrossEntropy(input, target_label)
- 生成: CrossEntropy(input[:-1], input[1:]) （预测下一个token）

GPT2LMHeadModel内部已经实现了语言建模loss：
loss = model(input_ids, labels=labels).loss

注意：labels中-100的位置会被忽略（我们的文章部分）
""")

# ============================================================
# 第五部分：训练模型
# ============================================================
print("\n" + "=" * 70)
print("第五部分：训练模型")
print("=" * 70)

# 添加早停机制和模型保存
best_loss = float('inf')
no_improve_count = 0
patience = 5  # 早停耐心值：5轮没有改善就停止

print(f"\n[训练设置]")
print(f"   早停耐心值: {patience} 轮")
print(f"   将自动保存最佳模型")

for epoch in range(EPOCHS):
    print(f"\n🚀 Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 50)
    
    # 训练阶段
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    print(f"   📉 Train Loss: {avg_loss:.4f}")
    
    # 保存最佳模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve_count = 0
        print(f"   ✨ 最佳模型！Loss: {best_loss:.4f}")
        
        # 保存临时最佳模型
        temp_save_dir = './saved_models/chinese_summarization_gpt2_temp'
        os.makedirs(temp_save_dir, exist_ok=True)
        model.save_pretrained(temp_save_dir)
        tokenizer.save_pretrained(temp_save_dir)
    else:
        no_improve_count += 1
        print(f"   ⏳ 未改善 ({no_improve_count}/{patience})")
        
        # 早停检查
        if no_improve_count >= patience:
            print(f"\n🛑 早停触发！连续 {patience} 轮未改善，停止训练")
            break

print(f"\n🎉 训练完成！最佳 Loss: {best_loss:.4f}")

# 加载最佳模型继续后续步骤
if os.path.exists('./saved_models/chinese_summarization_gpt2_temp'):
    print(f"\n📂 加载训练过程中的最佳模型...")
    model = GPT2LMHeadModel.from_pretrained('./saved_models/chinese_summarization_gpt2_temp')
    model = model.to(device)
    print(f"   ✓ 已加载最佳模型")

# ============================================================
# 第六部分：文本生成 - 解码策略
# ============================================================
print("\n" + "=" * 70)
print("第六部分：文本生成与解码策略")
print("=" * 70)

print("""
【解码策略对比】

1. Greedy Search（贪心搜索）
   - 每步选择概率最高的token
   - 优点：简单快速
   - 缺点：容易重复，缺乏多样性

2. Beam Search（束搜索）
   - 每步保留top-k个候选序列
   - 优点：质量较高
   - 缺点：计算量大，可能过于"安全"

3. Sampling（采样）
   - 按概率分布随机采样
   - 加入temperature控制随机性
   - 优点：多样性高
   - 缺点：可能不连贯

4. Top-k / Top-p (Nucleus) Sampling
   - 只从概率最高的k个或累积概率p内的token采样
   - 平衡质量和多样性
""")

model.eval()

# 测试生成
test_articles = [item["article"] for item in test_data[:3]]

print("\n📝 测试生成:")
print("=" * 70)

for i, article in enumerate(test_articles, 1):
    print(f"\n[示例 {i}]")
    print(f"文章: {article[:60]}...")
    
    # 编码文章
    article_encoding = tokenizer(
        article + "\n摘要：",
        return_tensors='pt',
        truncation=True,
        max_length=200
    )
    
    input_ids = article_encoding['input_ids'].to(device)
    
    # 方法1: Greedy Search
    with torch.no_grad():
        greedy_output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 30,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    greedy_summary = tokenizer.decode(
        greedy_output[0][input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    print(f"\n   [Greedy] {greedy_summary}")
    
    # 方法2: Beam Search
    with torch.no_grad():
        beam_output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 30,
            num_beams=4,
            num_return_sequences=1,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    beam_summary = tokenizer.decode(
        beam_output[0][input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    print(f"   [Beam]   {beam_summary}")
    
    # 方法3: Temperature Sampling
    with torch.no_grad():
        sample_output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 30,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    sample_summary = tokenizer.decode(
        sample_output[0][input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    print(f"   [Sample] {sample_summary}")
    
    print("-" * 50)

# ============================================================
# 第七部分：保存模型
# ============================================================
print("\n" + "=" * 70)
print("第七部分：保存模型")
print("=" * 70)

save_dir = './saved_models/chinese_summarization_gpt2'
os.makedirs(save_dir, exist_ok=True)

print(f"\n💾 保存模型到: {save_dir}")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# 保存配置
import json
config_info = {
    'model_name': 'gpt2',
    'task': 'text_summarization',
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'final_loss': float(best_loss),
    'max_article_len': 200,
    'max_summary_len': 50
}

with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(config_info, f, ensure_ascii=False, indent=2)

print(f"   ✓ 模型保存完成")

# 清理临时模型目录 - Windows需要确保文件句柄已释放
import shutil
import gc

# 强制垃圾回收，释放文件句柄
gc.collect()

temp_dir = './saved_models/chinese_summarization_gpt2_temp'
if os.path.exists(temp_dir):
    try:
        shutil.rmtree(temp_dir, onerror=lambda fn, path, exc: None)
        print(f"   ✓ 清理临时文件")
    except Exception as e:
        print(f"   [警告] 清理临时文件失败（不影响主模型）: {e}")

# ============================================================
# 第八部分：新文章摘要
# ============================================================
print("\n" + "=" * 70)
print("第八部分：新文章摘要生成")
print("=" * 70)

# 加载模型
loaded_tokenizer = GPT2Tokenizer.from_pretrained(save_dir)
loaded_model = GPT2LMHeadModel.from_pretrained(save_dir)
loaded_model = loaded_model.to(device)
loaded_model.eval()

print("\n🔄 模型加载完成")

# 新文章
new_articles = [
    "腾讯公司于今日发布2023年第四季度财报，营收达到1551.96亿元，同比增长7%。其中，游戏业务营收占比超过30%，微信月活跃用户数突破13亿。公司CEO马化腾表示，将继续加大在人工智能领域的投入。",
    "中国科学院宣布，我国首颗太阳探测科学技术试验卫星\"羲和号\"近日成功发射。该卫星将实现国际首次太阳Hα波段光谱成像的空间探测，填补太阳爆发源区高质量观测数据的空白，为我国太阳物理研究提供重要数据支撑。",
    "国家统计局数据显示，2023年我国国内生产总值（GDP）达到1260582亿元，按不变价格计算，比上年增长5.2%。分季度看，一季度GDP同比增长4.5%，二季度增长6.3%，三季度增长4.9%，四季度增长5.2%。",
]

print("\n📝 新文章摘要:")
print("=" * 70)

for i, article in enumerate(new_articles, 1):
    print(f"\n[文章 {i}]")
    print(f"原文: {article[:80]}...")
    
    # 编码
    article_encoding = loaded_tokenizer(
        article + "\n摘要：",
        return_tensors='pt',
        truncation=True,
        max_length=200
    )
    input_ids = article_encoding['input_ids'].to(device)
    
    # 生成（使用Beam Search）
    with torch.no_grad():
        output = loaded_model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 30,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,  # 避免重复2-gram
            pad_token_id=loaded_tokenizer.eos_token_id,
            eos_token_id=loaded_tokenizer.eos_token_id
        )
    
    # 解码（只取生成部分）
    summary = loaded_tokenizer.decode(
        output[0][input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    print(f"\n[生成摘要] {summary}")
    print("-" * 50)

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("项目总结")
print("=" * 70)

print("""
✅ 本项目完成了：

1. 文本生成数据格式
   - 输入: 文章 + [SEP]
   - 输出: 摘要 + [EOS]
   - Labels: 文章部分mask为-100（不计算loss）

2. GPT2 自回归训练
   - 逐个token预测下一个
   - Loss只计算摘要部分
   - 最大化条件概率 P(摘要|文章)

3. 解码策略对比
   - Greedy: 快速但可能重复
   - Beam Search: 质量高但保守
   - Sampling: 多样但需要调参
   - Top-k/p: 平衡方案

4. 生成技巧
   - no_repeat_ngram_size: 避免重复
   - early_stopping: 提前结束
   - temperature: 控制随机性

🚀 进阶方向：
   1. 使用更大的中文GPT模型（如GPT-3.5、ChatGLM）
   2. 实现ROUGE评估指标
   3. 使用Seq2Seq架构（Encoder-Decoder）
   4. 尝试T5、BART等专门的摘要模型
   5. 使用更大规模的真实数据集（如LCSTS、NLPCC）
""")

print("\n" + "=" * 70)
print("中文文本摘要实战完成！")
print("=" * 70)
