library(ggplot2)
library(dplyr)
library(reshape2)
df<-read.csv("/Users/benweinstein/Documents/EvergladesWadingBird/bird_detector_paper/Figures/penguin_results_900.csv")
df <- df %>% dplyr::select(-X)  %>% filter(proportion %in% c(0,1)) %>% mutate(proportion=factor(proportion))
df %>% ggplot(.,aes(x=as.factor(proportion),y=recall, fill=pretrained)) + geom_boxplot() + theme_bw() + xlab("") + ylab("Recall")
ggsave("/Users/benweinstein/Dropbox/Weecology/bird_detector/Figures/penguin_boxplot.png",height=3,width=6)
mdf<- df %>%  group_by(proportion,pretrained) %>% summarize(annotations = mean(annotations), mean=mean(recall), min=min(recall), max=max(recall))
ggplot(data=mdf, aes(x=proportion,y=mean, col=pretrained)) +
  geom_point(size=3) + geom_line(linetype="dashed",aes(y=mean)) +
  geom_errorbar(aes(ymin=min, ymax=max), width=0.2) + theme_bw() + xlab("Proportion of training data") + xlim(0,1) + ylab("Recall")
ggsave("/Users/benweinstein/Dropbox/Weecology/bird_detector/Figures/Penguin_recall.png",height=3, width=7)

box_data <- df %>% melt(id.vars=c("proportion","pretrained")) %>% filter(variable %in% c("recall")) %>%
  filter(!(proportion == 0 & pretrained=='False'))
line_data <- box_data %>% filter((proportion == 0 & pretrained=='True')) %>% distinct()
ggplot(box_data,aes(x=as.factor(pretrained),y=value),) + geom_boxplot( fill="grey50") +
  theme_bw() + xlab("Everglades Pretraining") + ylab("value") + geom_hline(data=line_data,aes(yintercept=value), linetype="dashed", col='red') + ylab("Penguin Recall")
ggsave("/Users/benweinstein/Dropbox/Weecology/bird_detector/Figures/Antarctic_results.png",height=3,width=8)

ggplot(df,aes(x=proportion,y=recall,col=pretrained)) + geom_point() +
  stat_smooth(
method="glm",
method.args=list(family="binomial"))

df<-read.csv("/Users/benweinstein/Documents/EvergladesWadingBird/bird_detector_paper/Figures/Palmyra_results_2500.csv")
head(df)
df <- df %>% dplyr::select(-X)  %>% filter(proportion %in% c(0,1)) %>% mutate(proportion=factor(proportion))
levels(df$proportion)<-c(FALSE, TRUE)
df %>% ggplot(.,aes(col=as.factor(proportion),x=recall, y=precision, shape=pretrained)) + geom_point(size=3) + theme_bw() + xlab("Recall") + ylab("Precision") + labs(shape="Everglades Pretrained", col="Local Training")

box_data <- df %>% melt(id.vars=c("proportion","pretrained")) %>% filter(variable %in% c("precision","recall")) %>%
  filter(!(proportion == FALSE & pretrained=='False'))
line_data <- box_data %>% filter((proportion == FALSE & pretrained=='True')) %>% distinct()
  ggplot(box_data,aes(x=as.factor(pretrained),y=value),) + geom_boxplot( fill="grey50") +
  theme_bw() + xlab("Everglades Pretraining") + ylab("value") + facet_wrap(~variable) + geom_hline(data=line_data,aes(yintercept=value), linetype="dashed", col='red')
ggsave("/Users/benweinstein/Dropbox/Weecology/bird_detector/Figures/Palmyra_results.png",height=3,width=8)
