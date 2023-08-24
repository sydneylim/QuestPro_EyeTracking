rm(list = ls())

library(tidyverse) #add column
library(plyr)
library(dplyr)
library(ggplot2)
library(ggsignif)
library(ggpubr)
library(rstatix)
library(emmeans)
library(phia)       #testInteractions
library(tidyr)      #spread
library(ARTool)     #art, artlm
library(coin)
library(DescTools)


#average errors: session 1 recalibrated with calibration task
average_session1_calibration_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session1_recalibrated_calibration_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')
#average errors: session 2 recalibrated with calibration task
average_session2_calibration_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session2_recalibrated_calibration_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')

#add column to specify trial
average_session1_calibration_data <- average_session1_calibration_data %>% add_column(trial = 1)
average_session2_calibration_data <- average_session2_calibration_data %>% add_column(trial = 2)

#turn variables into factors
average_session1_calibration_data$taskname = factor(average_session1_calibration_data$taskname)
average_session1_calibration_data$hololens = factor(average_session1_calibration_data$hololens)
average_session1_calibration_data$trial = factor(average_session1_calibration_data$trial)
average_session1_calibration_data$id = factor(average_session1_calibration_data$id)
average_session2_calibration_data$taskname = factor(average_session2_calibration_data$taskname)
average_session2_calibration_data$hololens = factor(average_session2_calibration_data$hololens)
average_session2_calibration_data$trial = factor(average_session2_calibration_data$trial)
average_session2_calibration_data$id = factor(average_session2_calibration_data$id)

#merge dataframes
total_calibration_error = rbind(average_session1_calibration_data, average_session2_calibration_data)

#plot between hololenses
ggplot(total_calibration_error, aes(x=taskname, y=cosineError, fill=hololens)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained Static', 'Head Constrained Moving', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Cosine Error (degrees)') + scale_fill_discrete(name="HoloLens")

#Three-way ANOVA: determine that HoloLens doesn't make a difference
total_calibration_error %>% group_by(taskname, trial, hololens) %>% shapiro_test(cosineError) 
m_hololens <- art(data = total_calibration_error, cosineError ~ taskname * trial * hololens + Error(id))
anova(m_hololens)

#plot between trials
ggplot(total_calibration_error, aes(x=taskname, y=cosineError, fill=trial)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained Static', 'Head Constrained Moving', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Cosine Error (degrees)') + scale_fill_discrete(name="", breaks=c("1", "2"), labels=c("Trial 1", "Trial 2"))

ggplot(total_calibration_error, aes(x=taskname, y=cosineError)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean", fill="lightblue") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained Static', 'Head Constrained Moving', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Cosine Error (degrees)') + scale_fill_discrete(name="", breaks=c("1", "2"), labels=c("Trial 1", "Trial 2"))


#Three-way ANOVA: determine if trial makes a difference
total_calibration_error %>% group_by(taskname, trial) %>% shapiro_test(cosineError) #fails test for normality, can't use normal ANOVA
m_total_cosine <- art(data = total_calibration_error, cosineError ~ taskname * trial + Error(id))
anova(m_total_cosine)

total_cosine_posthoc <- total_calibration_error %>% group_by(taskname) %>% rstatix::wilcox_test(cosineError ~ trial, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
total_cosine_posthoc

#separate calibrated and recalibrated Euclidean error
long_1_calibration_error <- average_session1_calibration_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')
long_2_calibration_error <- average_session2_calibration_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')

long_1_calibration_error$ErrorType = factor(long_1_calibration_error$ErrorType)
long_2_calibration_error$ErrorType = factor(long_2_calibration_error$ErrorType)

#graph calibrated and recalibrated Euclidean error
ggplot(long_1_calibration_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Device Calibration', 'Recalibration'))
ggplot(long_2_calibration_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Device Calibration', 'Recalibration'))

total_long_calibration_error = rbind(long_1_calibration_error, long_2_calibration_error)
ggplot(total_long_calibration_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained Static', 'Head Constrained Moving', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 50, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Device Calibration', 'Recalibration'))


#Three-way ART ANOVA: task, calibration, trial
total_long_calibration_error %>% group_by(taskname, trial, ErrorType) %>% shapiro_test(ErrorValue)
m_calibration <- art(data = total_long_calibration_error, ErrorValue ~ taskname * trial * ErrorType + Error(id))
anova(m_calibration)

m_calibration <- art(data = total_long_calibration_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_calibration)

calibration_errorType_posthoc <- total_long_calibration_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
calibration_errorType_posthoc

calibration_trial_posthoc <- total_long_calibration_error %>% group_by(taskname, ErrorType) %>% rstatix::wilcox_test(ErrorValue ~ trial, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
calibration_trial_posthoc

calibration_errorTypeByTrial_posthoc <- total_long_calibration_error %>% group_by(taskname, trial) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
calibration_errorTypeByTrial_posthoc

#plot total cosine error using Satyam's naming convention
ggplot(total_calibration_error, aes(x=taskname, y=cosineError)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean", fill="lightblue") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('ssHeadConstrained', 'wsBodyConstrained', 'wsWalking', 'ssWalking', 'hallway', 'calibration'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Recalibrate', 'w1', 'w2', 's4', 'w3', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Cosine Error (degrees)') + scale_fill_discrete(name="", breaks=c("1", "2"), labels=c("Trial 1", "Trial 2"))
#plot total uncalibrated Euclidean error using Satyam's naming convention
ggplot(total_calibration_error, aes(x=taskname, y=euclideanError)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean", fill="lightblue") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('ssHeadConstrained', 'wsBodyConstrained', 'wsWalking', 'ssWalking', 'hallway', 'calibration'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Recalibrate', 'w1', 'w2', 's4', 'w3', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="", breaks=c("1", "2"), labels=c("Trial 1", "Trial 2"))
#plot recalibrated error using Satyam's naming convention
ggplot(total_long_calibration_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('ssHeadConstrained', 'wsBodyConstrained', 'wsWalking', 'ssWalking', 'hallway', 'calibration'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Recalibrate', 'w1', 'w2', 's4', 'w3', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="", breaks=c("1", "2"), labels=c("Trial 1", "Trial 2"))


#-----------------------------------------------------------------------------------------

rm(list = ls())
#average errors: session 1 recalibrated with ssHeadConstrained task
average_session1_ssHeadConstrained_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session1_recalibrated_ssHeadConstrained_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')
#average errors: session 2 recalibrated with ssHeadConstrained task
average_session2_ssHeadConstrained_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session2_recalibrated_ssHeadConstrained_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')

#add column to specify trial
average_session1_ssHeadConstrained_data <- average_session1_ssHeadConstrained_data %>% add_column(trial = '1')
average_session2_ssHeadConstrained_data <- average_session2_ssHeadConstrained_data %>% add_column(trial = '2')

#turn variables into factors
average_session1_ssHeadConstrained_data$taskname = factor(average_session1_ssHeadConstrained_data$taskname)
average_session1_ssHeadConstrained_data$hololens = factor(average_session1_ssHeadConstrained_data$hololens)
average_session1_ssHeadConstrained_data$trial = factor(average_session1_ssHeadConstrained_data$trial)
average_session1_ssHeadConstrained_data$id = factor(average_session1_ssHeadConstrained_data$id)
average_session2_ssHeadConstrained_data$taskname = factor(average_session2_ssHeadConstrained_data$taskname)
average_session2_ssHeadConstrained_data$hololens = factor(average_session2_ssHeadConstrained_data$hololens)
average_session2_ssHeadConstrained_data$trial = factor(average_session2_ssHeadConstrained_data$trial)
average_session2_ssHeadConstrained_data$id = factor(average_session2_ssHeadConstrained_data$id)

#split by recalibrated and uncalibrated Euclidean error
long_average_session1_ssHeadConstrained_data <- average_session1_ssHeadConstrained_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue') 
long_average_session2_ssHeadConstrained_data <- average_session2_ssHeadConstrained_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')

#turn error types into factors
long_average_session1_ssHeadConstrained_data$ErrorType = factor(long_average_session1_ssHeadConstrained_data$ErrorType)
long_average_session2_ssHeadConstrained_data$ErrorType = factor(long_average_session2_ssHeadConstrained_data$ErrorType)

#plot calibrated and uncalibrated error separated by trial
ggplot(long_average_session1_ssHeadConstrained_data, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))
ggplot(long_average_session2_ssHeadConstrained_data, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))

#merge dataframes
total_ssHeadConstrained_error = rbind(long_average_session1_ssHeadConstrained_data, long_average_session2_ssHeadConstrained_data)
ggplot(total_ssHeadConstrained_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained Static', 'Head Constrained Moving', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 50, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Device Calibration', 'Recalibration'))

#Three-way ANOVA: task, error type, trial
total_ssHeadConstrained_error %>% group_by(taskname, trial, ErrorType) %>% shapiro_test(ErrorValue) #fails test for normality, can't use normal ANOVA

m_ssHeadConstrained = art(data = total_ssHeadConstrained_error, ErrorValue ~ taskname * trial * ErrorType + Error(id))
anova(m_ssHeadConstrained)

m_ssHeadConstrained = art(data = total_ssHeadConstrained_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_ssHeadConstrained)

ssHeadConstrained_error_posthoc <- total_ssHeadConstrained_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
ssHeadConstrained_error_posthoc

ssHeadConstrained_learning_effect_posthoc <- total_ssHeadConstrained_error %>% group_by(taskname, ErrorType) %>% rstatix::wilcox_test(ErrorValue ~ trial, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
ssHeadConstrained_learning_effect_posthoc

ssHeadConstrained_by_trial_error_posthoc <- total_ssHeadConstrained_error %>% group_by(taskname, trial) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
ssHeadConstrained_by_trial_error_posthoc




#-----------------------------------------------------------------------------------------

rm(list = ls())
#average errors: session 1 recalibrated with wsBodyConstrained task
average_session1_wsBodyConstrained_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session1_recalibrated_wsBodyConstrained_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')
#average errors: session 2 recalibrated with wsBodyConstrained task
average_session2_wsBodyConstrained_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session2_recalibrated_wsBodyConstrained_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')

#add column to specify trial
average_session1_wsBodyConstrained_data <- average_session1_wsBodyConstrained_data %>% add_column(trial = '1')
average_session2_wsBodyConstrained_data <- average_session2_wsBodyConstrained_data %>% add_column(trial = '2')

#turn variables into factors
average_session1_wsBodyConstrained_data$taskname = factor(average_session1_wsBodyConstrained_data$taskname)
average_session1_wsBodyConstrained_data$hololens = factor(average_session1_wsBodyConstrained_data$hololens)
average_session1_wsBodyConstrained_data$trial = factor(average_session1_wsBodyConstrained_data$trial)
average_session1_wsBodyConstrained_data$id = factor(average_session1_wsBodyConstrained_data$id)
average_session2_wsBodyConstrained_data$taskname = factor(average_session2_wsBodyConstrained_data$taskname)
average_session2_wsBodyConstrained_data$hololens = factor(average_session2_wsBodyConstrained_data$hololens)
average_session2_wsBodyConstrained_data$trial = factor(average_session2_wsBodyConstrained_data$trial)
average_session2_wsBodyConstrained_data$id = factor(average_session2_wsBodyConstrained_data$id)

#split by recalibrated and uncalibrated Euclidean error
long_average_session1_wsBodyConstrained_data <- average_session1_wsBodyConstrained_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue') 
long_average_session2_wsBodyConstrained_data <- average_session2_wsBodyConstrained_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')

#turn error types into factors
long_average_session1_wsBodyConstrained_data$ErrorType = factor(long_average_session1_wsBodyConstrained_data$ErrorType)
long_average_session2_wsBodyConstrained_data$ErrorType = factor(long_average_session2_wsBodyConstrained_data$ErrorType)

#plot calibrated and uncalibrated error separated by trial
ggplot(long_average_session1_wsBodyConstrained_data, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))
ggplot(long_average_session2_wsBodyConstrained_data, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))

#merge dataframes
total_wsBodyConstrained_error = rbind(long_average_session1_wsBodyConstrained_data, long_average_session2_wsBodyConstrained_data)
ggplot(total_wsBodyConstrained_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained Static', 'Head Constrained Moving', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 50, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Device Calibration', 'Recalibration'))

#Three-way ANOVA: task, error type, trial
total_wsBodyConstrained_error %>% group_by(taskname, trial, ErrorType) %>% shapiro_test(ErrorValue) #fails test for normality, can't use normal ANOVA

m_wsBodyConstrained = art(data = total_wsBodyConstrained_error, ErrorValue ~ taskname * trial * ErrorType + Error(id))
anova(m_wsBodyConstrained)

m_wsBodyConstrained = art(data = total_wsBodyConstrained_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_wsBodyConstrained)

wsBodyConstrained_error_posthoc <- total_wsBodyConstrained_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
wsBodyConstrained_error_posthoc

wsBodyConstrained_learning_effect_posthoc <- total_wsBodyConstrained_error %>% group_by(taskname, ErrorType) %>% rstatix::wilcox_test(ErrorValue ~ trial, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
wsBodyConstrained_learning_effect_posthoc

wsBodyConstrained_by_trial_error_posthoc <- total_wsBodyConstrained_error %>% group_by(taskname, trial) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
wsBodyConstrained_by_trial_error_posthoc





#-----------------------------------------------------------------------------------------

rm(list = ls())
#average errors: session 1 recalibrated with ssWalking task
average_session1_ssWalking_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session1_recalibrated_ssWalking_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')
#average errors: session 2 recalibrated with ssWalking task
average_session2_ssWalking_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session2_recalibrated_ssWalking_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')

#add column to specify trial
average_session1_ssWalking_data <- average_session1_ssWalking_data %>% add_column(trial = '1')
average_session2_ssWalking_data <- average_session2_ssWalking_data %>% add_column(trial = '2')

#turn variables into factors
average_session1_ssWalking_data$taskname = factor(average_session1_ssWalking_data$taskname)
average_session1_ssWalking_data$hololens = factor(average_session1_ssWalking_data$hololens)
average_session1_ssWalking_data$trial = factor(average_session1_ssWalking_data$trial)
average_session1_ssWalking_data$id = factor(average_session1_ssWalking_data$id)
average_session2_ssWalking_data$taskname = factor(average_session2_ssWalking_data$taskname)
average_session2_ssWalking_data$hololens = factor(average_session2_ssWalking_data$hololens)
average_session2_ssWalking_data$trial = factor(average_session2_ssWalking_data$trial)
average_session2_ssWalking_data$id = factor(average_session2_ssWalking_data$id)

#split by recalibrated and uncalibrated Euclidean error
long_average_session1_ssWalking_data <- average_session1_ssWalking_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue') 
long_average_session2_ssWalking_data <- average_session2_ssWalking_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')

#turn error types into factors
long_average_session1_ssWalking_data$ErrorType = factor(long_average_session1_ssWalking_data$ErrorType)
long_average_session2_ssWalking_data$ErrorType = factor(long_average_session2_ssWalking_data$ErrorType)

#plot calibrated and uncalibrated error separated by trial
ggplot(long_average_session1_ssWalking_data, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))
ggplot(long_average_session2_ssWalking_data, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))

#merge dataframes
total_ssWalking_error = rbind(long_average_session1_ssWalking_data, long_average_session2_ssWalking_data)
ggplot(total_ssWalking_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained Static', 'Head Constrained Moving', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 50, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Device Calibration', 'Recalibration'))

#Three-way ANOVA: task, error type, trial
total_ssWalking_error %>% group_by(taskname, trial, ErrorType) %>% shapiro_test(ErrorValue) #fails test for normality, can't use normal ANOVA

m_ssWalking = art(data = total_ssWalking_error, ErrorValue ~ taskname * trial * ErrorType + Error(id))
anova(m_ssWalking)

m_ssWalking = art(data = total_ssWalking_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_ssWalking)

ssWalking_error_posthoc <- total_ssWalking_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
ssWalking_error_posthoc

ssWalking_learning_effect_posthoc <- total_ssWalking_error %>% group_by(taskname, ErrorType) %>% rstatix::wilcox_test(ErrorValue ~ trial, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
ssWalking_learning_effect_posthoc

ssWalking_by_trial_error_posthoc <- total_ssWalking_error %>% group_by(taskname, trial) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
ssWalking_by_trial_error_posthoc

#-----------------------------------------------------------------------------------------

rm(list = ls())
#average errors: session 1 recalibrated with wsWalking task
average_session1_wsWalking_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session1_recalibrated_wsWalking_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')
#average errors: session 2 recalibrated with wsWalking task
average_session2_wsWalking_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session2_recalibrated_wsWalking_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')

#add column to specify trial
average_session1_wsWalking_data <- average_session1_wsWalking_data %>% add_column(trial = '1')
average_session2_wsWalking_data <- average_session2_wsWalking_data %>% add_column(trial = '2')

#turn variables into factors
average_session1_wsWalking_data$taskname = factor(average_session1_wsWalking_data$taskname)
average_session1_wsWalking_data$hololens = factor(average_session1_wsWalking_data$hololens)
average_session1_wsWalking_data$trial = factor(average_session1_wsWalking_data$trial)
average_session1_wsWalking_data$id = factor(average_session1_wsWalking_data$id)
average_session2_wsWalking_data$taskname = factor(average_session2_wsWalking_data$taskname)
average_session2_wsWalking_data$hololens = factor(average_session2_wsWalking_data$hololens)
average_session2_wsWalking_data$trial = factor(average_session2_wsWalking_data$trial)
average_session2_wsWalking_data$id = factor(average_session2_wsWalking_data$id)

#split by recalibrated and uncalibrated Euclidean error
long_average_session1_wsWalking_data <- average_session1_wsWalking_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue') 
long_average_session2_wsWalking_data <- average_session2_wsWalking_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')

#turn error types into factors
long_average_session1_wsWalking_data$ErrorType = factor(long_average_session1_wsWalking_data$ErrorType)
long_average_session2_wsWalking_data$ErrorType = factor(long_average_session2_wsWalking_data$ErrorType)

#plot calibrated and uncalibrated error separated by trial
ggplot(long_average_session1_wsWalking_data, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))
ggplot(long_average_session2_wsWalking_data, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))

#merge dataframes
total_wsWalking_error = rbind(long_average_session1_wsWalking_data, long_average_session2_wsWalking_data)
ggplot(total_wsWalking_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained Static', 'Head Constrained Moving', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 50, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Device Calibration', 'Recalibration'))

#Three-way ANOVA: task, error type, trial
total_wsWalking_error %>% group_by(taskname, trial, ErrorType) %>% shapiro_test(ErrorValue) #fails test for normality, can't use normal ANOVA

m_wsWalking = art(data = total_wsWalking_error, ErrorValue ~ taskname * trial * ErrorType + Error(id))
anova(m_wsWalking)

m_wsWalking = art(data = total_wsWalking_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_wsWalking)

wsWalking_error_posthoc <- total_wsWalking_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
wsWalking_error_posthoc

wsWalking_learning_effect_posthoc <- total_wsWalking_error %>% group_by(taskname, ErrorType) %>% rstatix::wilcox_test(ErrorValue ~ trial, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
wsWalking_learning_effect_posthoc

wsWalking_by_trial_error_posthoc <- total_wsWalking_error %>% group_by(taskname, trial) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
wsWalking_by_trial_error_posthoc



#-----------------------------------------------------------------------------------------

rm(list = ls())
#average errors: session 1 recalibrated with hallway task
average_session1_hallway_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session1_recalibrated_hallway_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')
#average errors: session 2 recalibrated with hallway task
average_session2_hallway_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session2_recalibrated_hallway_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')

#add column to specify trial
average_session1_hallway_data <- average_session1_hallway_data %>% add_column(trial = '1')
average_session2_hallway_data <- average_session2_hallway_data %>% add_column(trial = '2')

#turn variables into factors
average_session1_hallway_data$taskname = factor(average_session1_hallway_data$taskname)
average_session1_hallway_data$hololens = factor(average_session1_hallway_data$hololens)
average_session1_hallway_data$trial = factor(average_session1_hallway_data$trial)
average_session1_hallway_data$id = factor(average_session1_hallway_data$id)
average_session2_hallway_data$taskname = factor(average_session2_hallway_data$taskname)
average_session2_hallway_data$hololens = factor(average_session2_hallway_data$hololens)
average_session2_hallway_data$trial = factor(average_session2_hallway_data$trial)
average_session2_hallway_data$id = factor(average_session2_hallway_data$id)

#split by recalibrated and uncalibrated Euclidean error
long_average_session1_hallway_data <- average_session1_hallway_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue') 
long_average_session2_hallway_data <- average_session2_hallway_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')

#turn error types into factors
long_average_session1_hallway_data$ErrorType = factor(long_average_session1_hallway_data$ErrorType)
long_average_session2_hallway_data$ErrorType = factor(long_average_session2_hallway_data$ErrorType)

#plot calibrated and uncalibrated error separated by trial
ggplot(long_average_session1_hallway_data, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))
ggplot(long_average_session2_hallway_data, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))

#merge dataframes
total_hallway_error = rbind(long_average_session1_hallway_data, long_average_session2_hallway_data)
ggplot(total_hallway_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained Static', 'Head Constrained Moving', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Device Calibration', 'Recalibration'))

#Three-way ANOVA: task, error type, trial
total_hallway_error %>% group_by(taskname, trial, ErrorType) %>% shapiro_test(ErrorValue) #fails test for normality, can't use normal ANOVA

m_hallway = art(data = total_hallway_error, ErrorValue ~ taskname * trial * ErrorType + Error(id))
anova(m_hallway)

m_hallway = art(data = total_hallway_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_hallway)

hallway_error_posthoc <- total_hallway_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
hallway_error_posthoc

hallway_learning_effect_posthoc <- total_hallway_error %>% group_by(taskname, ErrorType) %>% rstatix::wilcox_test(ErrorValue ~ trial, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
hallway_learning_effect_posthoc

hallway_by_trial_error_posthoc <- total_hallway_error %>% group_by(taskname, trial) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
hallway_by_trial_error_posthoc


#-----------------------------------------------------------------------------------------

rm(list = ls())
#average errors: session 1 moving
average_session1_moving_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session1_moving_recalibrated_calibration_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')
#average errors: session 1 static
average_session1_static_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session1_static_recalibrated_calibration_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')
#average errors: session 2 moving
average_session2_moving_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session2_moving_recalibrated_calibration_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')
#average errors: session 2 static
average_session2_static_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/session2_static_recalibrated_calibration_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')


#add column to specify trial
average_session1_moving_data <- average_session1_moving_data %>% add_column(trial = '1')
average_session1_static_data <- average_session1_static_data %>% add_column(trial = '1')

average_session2_moving_data <- average_session2_moving_data %>% add_column(trial = '2')
average_session2_static_data <- average_session2_static_data %>% add_column(trial = '2')

average_session1_moving_data <- average_session1_moving_data %>% add_column(movement = 'Moving')
average_session1_static_data <- average_session1_static_data %>% add_column(movement = 'Static')

average_session2_moving_data <- average_session2_moving_data %>% add_column(movement = 'Moving')
average_session2_static_data <- average_session2_static_data %>% add_column(movement = 'Static')

#turn variables into factors
average_session1_moving_data$taskname = factor(average_session1_moving_data$taskname)
average_session1_moving_data$hololens = factor(average_session1_moving_data$hololens)
average_session1_moving_data$trial = factor(average_session1_moving_data$trial)
average_session1_moving_data$id = factor(average_session1_moving_data$id)
average_session1_moving_data$movement = factor(average_session1_moving_data$movement)

average_session1_static_data$taskname = factor(average_session1_static_data$taskname)
average_session1_static_data$hololens = factor(average_session1_static_data$hololens)
average_session1_static_data$trial = factor(average_session1_static_data$trial)
average_session1_static_data$id = factor(average_session1_static_data$id)
average_session1_static_data$movement = factor(average_session1_static_data$movement)

average_session2_moving_data$taskname = factor(average_session2_moving_data$taskname)
average_session2_moving_data$hololens = factor(average_session2_moving_data$hololens)
average_session2_moving_data$trial = factor(average_session2_moving_data$trial)
average_session2_moving_data$id = factor(average_session2_moving_data$id)
average_session2_moving_data$movement = factor(average_session2_moving_data$movement)

average_session2_static_data$taskname = factor(average_session2_static_data$taskname)
average_session2_static_data$hololens = factor(average_session2_static_data$hololens)
average_session2_static_data$trial = factor(average_session2_static_data$trial)
average_session2_static_data$id = factor(average_session2_static_data$id)
average_session2_static_data$movement = factor(average_session2_static_data$movement)

#split by recalibrated and uncalibrated Euclidean error
long_average_session1_moving_data <- average_session1_moving_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue') 
long_average_session1_static_data <- average_session1_static_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue') 
long_average_session2_moving_data <- average_session2_moving_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')
long_average_session2_static_data <- average_session2_static_data |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')

#turn error types into factors
long_average_session1_moving_data$ErrorType = factor(long_average_session1_moving_data$ErrorType)
long_average_session1_static_data$ErrorType = factor(long_average_session1_static_data$ErrorType)
long_average_session2_moving_data$ErrorType = factor(long_average_session2_moving_data$ErrorType)
long_average_session2_static_data$ErrorType = factor(long_average_session2_static_data$ErrorType)

#combine sessions
long_session1_movement_data = rbind(long_average_session1_moving_data, long_average_session1_static_data)
long_session2_movement_data = rbind(long_average_session2_moving_data, long_average_session2_static_data)
session1_movement_data = rbind(average_session1_moving_data, average_session1_static_data)
session2_movement_data = rbind(average_session2_moving_data, average_session2_static_data)

#remove calibration task
long_session1_movement_data <- long_session1_movement_data[long_session1_movement_data$taskname != 'calibration',]
long_session2_movement_data <- long_session2_movement_data[long_session2_movement_data$taskname != 'calibration',]
session1_movement_data <- session1_movement_data[session1_movement_data$taskname != 'calibration',]
session2_movement_data <- session2_movement_data[session2_movement_data$taskname != 'calibration',]

#plot cosine error
ggplot(session1_movement_data, aes(x=taskname, y=cosineError, fill=movement)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Cosine Error (degrees)') + scale_fill_discrete(name="")
ggplot(session2_movement_data, aes(x=taskname, y=cosineError, fill=movement)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Cosine Error (degrees)') + scale_fill_discrete(name="")

#plot uncalibrated Euclidean error
ggplot(session1_movement_data, aes(x=taskname, y=euclideanError, fill=movement)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="")
ggplot(session2_movement_data, aes(x=taskname, y=euclideanError, fill=movement)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="")

#Three-way ANOVA: task, trial, movement
movement_error = rbind(session1_movement_data, session2_movement_data)
ggplot(movement_error, aes(x=taskname, y=cosineError, fill=movement)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained Moving', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 50, vjust = 1, hjust=1)) + ylab('Cosine Error (degrees)') + scale_fill_discrete(name="")


movement_error %>% group_by(taskname, trial, movement) %>% shapiro_test(cosineError) #fails test for normality, can't use normal ANOVA

m_movement = art(data = movement_error, cosineError ~ taskname * trial * movement + Error(id))
anova(m_movement)

movement_error_posthoc <- movement_error %>% group_by(taskname) %>% rstatix::wilcox_test(cosineError ~ movement, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
movement_error_posthoc

movement_learning_effect_posthoc <- movement_error %>% group_by(taskname, movement) %>% rstatix::wilcox_test(cosineError ~ trial, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
movement_learning_effect_posthoc


#plot graph with same format as Satyam's
both_movement <- data.frame(movement_error)
both_movement$movement = 'Both'

graph_data <- rbind(movement_error, both_movement)
ggplot(graph_data, aes(x=taskname, y=cosineError, fill=movement)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('ssHeadConstrained', 'wsBodyConstrained', 'wsWalking', 'ssWalking', 'hallway'), breaks=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('w1', 'w2', 's4', 'w3', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Cosine Error (degrees)') + scale_fill_discrete(name="")
ggplot(movement_error, aes(x=taskname, y=cosineError, fill=movement)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('ssHeadConstrained', 'wsBodyConstrained', 'wsWalking', 'ssWalking', 'hallway'), breaks=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('w1', 'w2', 's4', 'w3', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Cosine Error (degrees)') + scale_fill_discrete(name="")


##########################################################################################
#OLD STUFF, SEPARATED BY HOLOLENS
#average errors: recalibrated with calibration task
average_error_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/recalibrated_calibration_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')

#turn variables into factors
average_error_data$taskname = factor(average_error_data$taskname)
average_error_data$hololens = factor(average_error_data$hololens)
average_error_data$id = factor(average_error_data$id)

#separate between HoloLenses
a_average_error <- average_error_data[average_error_data$hololens=='A',]
b_average_error <- average_error_data[average_error_data$hololens=='B',]

#change dataframes into graphable format
long_a_average_error <- a_average_error |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')
long_b_average_error <- b_average_error |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')

long_a_average_error$ErrorType = factor(long_a_average_error$ErrorType)
long_a_average_error$taskname = factor(long_a_average_error$taskname)
long_b_average_error$ErrorType = factor(long_b_average_error$ErrorType)
long_b_average_error$taskname = factor(long_b_average_error$taskname)

#plot dataframes for cosine error with hololenses
ggplot(average_error_data, aes(x=taskname, y=cosineError, fill=hololens)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Cosine Error (degrees)') + scale_fill_discrete(name="HoloLens")


#plot dataframes for calibrated and uncalibrated euclidean error with 95% confidence interval
ggplot(long_a_average_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))
ggplot(long_b_average_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))

#plot dataframes for local cosine error
ggplot(a_average_error, aes(x=taskname, y=cosineError)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean", fill='goldenrod') + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Cosine Error (degrees)')
ggplot(b_average_error, aes(x=taskname, y=cosineError)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean", fill='goldenrod') + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Cosine Error (degrees)')


#One way repeated measures (within subjects) ANOVA: find differences in accuracy between tasks
#result <- aov(cosineError ~ taskname + Error(id), data = a_average_error)
#summary(result)
#attach(a_average_error)
#posthoc <- pairwise.t.test(cosineError,taskname, p.adj = "bonf", paired = TRUE)
#posthoc
#detach(a_average_error)
#posthoc <- a_average_error %>% pairwise_t_test(cosineError ~ taskname, paired = TRUE, p.adjust.method = "bonferroni")
#posthoc

#result <- aov(cosineError ~ taskname + Error(id), data = b_average_error)
#summary(result)
#attach(b_average_error)
#posthoc <- pairwise.t.test(cosineError,taskname, p.adj = "bonf", paired = TRUE)
#posthoc
#detach(b_average_error)

a_average_error %>% group_by(taskname) %>% shapiro_test(cosineError) #fails test for normality, can't use normal ANOVA
m_average_a_error <- art(data = a_average_error, cosineError ~ taskname + Error(id))
anova(m_average_a_error)

a_average_error_posthoc <- a_average_error %>% rstatix::wilcox_test(cosineError ~ taskname, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
a_average_error_posthoc



b_average_error %>% group_by(taskname) %>% shapiro_test(cosineError) #fails test for normality, can't use normal ANOVA
m_average_b_error <- art(data = b_average_error, cosineError ~ taskname + Error(id))
anova(m_average_b_error)

b_average_error_posthoc <- b_average_error %>% rstatix::wilcox_test(cosineError ~ taskname, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
b_average_error_posthoc

#Two way mixed ANOVA: determine if headset makes a difference
#https://www.datanovia.com/en/lessons/mixed-anova-in-r/
#result <- anova_test(data = average_error_data, dv = cosineError, wid=id, between=hololens, within=taskname)
#get_anova_table(result)

average_error_data %>% group_by(taskname, hololens) %>% shapiro_test(cosineError) #fails test for normality, can't use normal ANOVA
m_average_hololens_error <- art(data = average_error_data, cosineError ~ taskname * hololens + Error(id))
anova(m_average_hololens_error)


#Two way ANOVA for calibration
#result <- aov(ErrorValue ~ taskname * ErrorType + Error(id/(taskname * ErrorType)), data = long_a_average_error)
#summary(result)

#model <- art(data = long_a_average_error, ErrorValue ~ taskname * ErrorType + Error(id))
#anova(model)

#posthoc <- long_a_average_error %>% group_by(taskname) %>% pairwise_t_test(ErrorValue ~ ErrorType, paired = TRUE, p.adjust.method = "bonferroni")
#posthoc

long_a_average_error %>% group_by(taskname, ErrorType) %>% shapiro_test(ErrorValue)
m_a_calibration_error <- art(data = long_a_average_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_a_calibration_error)

a_calibration_posthoc <- long_a_average_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
a_calibration_posthoc



long_b_average_error %>% group_by(taskname, ErrorType) %>% shapiro_test(ErrorValue)
m_b_calibration_error <- art(data = long_b_average_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_b_calibration_error)

#result <- aov(ErrorValue ~ taskname * ErrorType + Error(id/(taskname * ErrorType)), data = long_b_average_error)
#summary(result)









#average errors: recalibrated with ssHeadConstrained task
average_error_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/recalibrated_ssHeadConstrained_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')

#turn variables into factors
average_error_data$taskname = factor(average_error_data$taskname)
average_error_data$hololens = factor(average_error_data$hololens)
average_error_data$id = factor(average_error_data$id)

#separate between HoloLenses
a_average_error <- average_error_data[average_error_data$hololens=='A',]
b_average_error <- average_error_data[average_error_data$hololens=='B',]

#change dataframes into graphable format
long_a_average_error <- a_average_error |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')
long_b_average_error <- b_average_error |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')

long_a_average_error$ErrorType = factor(long_a_average_error$ErrorType)
long_a_average_error$taskname = factor(long_a_average_error$taskname)
long_b_average_error$ErrorType = factor(long_b_average_error$ErrorType)
long_b_average_error$taskname = factor(long_b_average_error$taskname)

#plot dataframes for calibrated and uncalibrated euclidean error with 95% confidence interval
ggplot(long_a_average_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))
ggplot(long_b_average_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))

#Two way ANOVA
#result <- aov(ErrorValue ~ taskname * ErrorType  + Error(id/(taskname * ErrorType)), data = long_a_average_error)
#summary(result)
#posthoc <- long_a_average_error %>% group_by(taskname) %>% pairwise_t_test(ErrorValue ~ ErrorType, paired = TRUE, p.adjust.method = "bonferroni")
#posthoc

#result <- aov(ErrorValue ~ taskname * ErrorType + Error(id/(taskname * ErrorType)), data = long_b_average_error)
#summary(result)

long_a_average_error %>% group_by(taskname, ErrorType) %>% shapiro_test(ErrorValue)
m_a_calibration_error <- art(data = long_a_average_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_a_calibration_error)

a_calibration_posthoc <- long_a_average_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
a_calibration_posthoc



long_b_average_error %>% group_by(taskname, ErrorType) %>% shapiro_test(ErrorValue)
m_b_calibration_error <- art(data = long_b_average_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_b_calibration_error)







#average errors: recalibrated with wsBodyConstrained task
average_error_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/recalibrated_wsBodyConstrained_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')

#turn variables into factors
average_error_data$taskname = factor(average_error_data$taskname)
average_error_data$hololens = factor(average_error_data$hololens)
average_error_data$id = factor(average_error_data$id)

#separate between HoloLenses
a_average_error <- average_error_data[average_error_data$hololens=='A',]
b_average_error <- average_error_data[average_error_data$hololens=='B',]

#change dataframes into graphable format
long_a_average_error <- a_average_error |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')
long_b_average_error <- b_average_error |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')

long_a_average_error$ErrorType = factor(long_a_average_error$ErrorType)
long_b_average_error$ErrorType = factor(long_b_average_error$ErrorType)

#plot dataframes for calibrated and uncalibrated euclidean error with 95% confidence interval
ggplot(long_a_average_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))
ggplot(long_b_average_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))

#Two way ANOVA
#result <- aov(ErrorValue ~ taskname * ErrorType + Error(id/(taskname * ErrorType)), data = long_a_average_error)
#summary(result)
#posthoc <- long_a_average_error %>% group_by(taskname) %>% pairwise_t_test(ErrorValue ~ ErrorType, paired = TRUE, p.adjust.method = "bonferroni")
#posthoc

#result <- aov(ErrorValue ~ taskname * ErrorType + Error(id/(taskname * ErrorType)), data = long_b_average_error)
#summary(result)

long_a_average_error %>% group_by(taskname, ErrorType) %>% shapiro_test(ErrorValue)
m_a_calibration_error <- art(data = long_a_average_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_a_calibration_error)

a_calibration_posthoc <- long_a_average_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
a_calibration_posthoc


long_b_average_error %>% group_by(taskname, ErrorType) %>% shapiro_test(ErrorValue)
m_b_calibration_error <- art(data = long_b_average_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_b_calibration_error)

b_calibration_posthoc <- long_b_average_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
b_calibration_posthoc






#average errors: recalibrated with ssWalking task
average_error_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/recalibrated_ssWalking_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')

#turn variables into factors
average_error_data$taskname = factor(average_error_data$taskname)
average_error_data$hololens = factor(average_error_data$hololens)
average_error_data$id = factor(average_error_data$id)

#separate between HoloLenses
a_average_error <- average_error_data[average_error_data$hololens=='A',]
b_average_error <- average_error_data[average_error_data$hololens=='B',]

#change dataframes into graphable format
long_a_average_error <- a_average_error |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')
long_b_average_error <- b_average_error |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')

long_a_average_error$ErrorType = factor(long_a_average_error$ErrorType)
long_b_average_error$ErrorType = factor(long_b_average_error$ErrorType)

#plot dataframes for calibrated and uncalibrated euclidean error with 95% confidence interval
ggplot(long_a_average_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))
ggplot(long_b_average_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))

#Two way ANOVA
#result <- aov(ErrorValue ~ taskname * ErrorType + Error(id/(taskname * ErrorType)), data = long_a_average_error)
#summary(result)
#posthoc <- long_a_average_error %>% group_by(taskname) %>% pairwise_t_test(ErrorValue ~ ErrorType, paired = TRUE, p.adjust.method = "bonferroni")
#posthoc

#result <- aov(ErrorValue ~ taskname * ErrorType + Error(id/(taskname * ErrorType)), data = long_b_average_error)
#summary(result)

long_a_average_error %>% group_by(taskname, ErrorType) %>% shapiro_test(ErrorValue)
m_a_calibration_error <- art(data = long_a_average_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_a_calibration_error)

a_calibration_posthoc <- long_a_average_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
a_calibration_posthoc


long_b_average_error %>% group_by(taskname, ErrorType) %>% shapiro_test(ErrorValue)
m_b_calibration_error <- art(data = long_b_average_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_b_calibration_error)

b_calibration_posthoc <- long_b_average_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
b_calibration_posthoc








#average errors: recalibrated with wsWalking task
average_error_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/recalibrated_wsWalking_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')

#turn variables into factors
average_error_data$taskname = factor(average_error_data$taskname)
average_error_data$hololens = factor(average_error_data$hololens)
average_error_data$id = factor(average_error_data$id)

#separate between HoloLenses
a_average_error <- average_error_data[average_error_data$hololens=='A',]
b_average_error <- average_error_data[average_error_data$hololens=='B',]

#change dataframes into graphable format
long_a_average_error <- a_average_error |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')
long_b_average_error <- b_average_error |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')

long_a_average_error$ErrorType = factor(long_a_average_error$ErrorType)
long_b_average_error$ErrorType = factor(long_b_average_error$ErrorType)

#plot dataframes for calibrated and uncalibrated euclidean error with 95% confidence interval
ggplot(long_a_average_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))
ggplot(long_b_average_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))

#Two way ANOVA
#result <- aov(ErrorValue ~ taskname * ErrorType + Error(id/(taskname * ErrorType)), data = long_a_average_error)
#summary(result)
#posthoc <- long_a_average_error %>% group_by(taskname) %>% pairwise_t_test(ErrorValue ~ ErrorType, paired = TRUE, p.adjust.method = "bonferroni")
#posthoc

#result <- aov(ErrorValue ~ taskname * ErrorType + Error(id/(taskname * ErrorType)), data = long_b_average_error)
#summary(result)

long_a_average_error %>% group_by(taskname, ErrorType) %>% shapiro_test(ErrorValue)
m_a_calibration_error <- art(data = long_a_average_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_a_calibration_error)

a_calibration_posthoc <- long_a_average_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
a_calibration_posthoc


long_b_average_error %>% group_by(taskname, ErrorType) %>% shapiro_test(ErrorValue)
m_b_calibration_error <- art(data = long_b_average_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_b_calibration_error)

b_calibration_posthoc <- long_b_average_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
b_calibration_posthoc






#average errors: recalibrated with hallway task
average_error_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/recalibrated_hallway_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')

#turn variables into factors
average_error_data$taskname = factor(average_error_data$taskname)
average_error_data$hololens = factor(average_error_data$hololens)
average_error_data$id = factor(average_error_data$id)

#separate between HoloLenses
a_average_error <- average_error_data[average_error_data$hololens=='A',]
b_average_error <- average_error_data[average_error_data$hololens=='B',]

#change dataframes into graphable format
long_a_average_error <- a_average_error |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')
long_b_average_error <- b_average_error |> pivot_longer(cols = c('euclideanError', 'recalibratedEuclideanError'), names_to = 'ErrorType', values_to = 'ErrorValue')

long_a_average_error$ErrorType = factor(long_a_average_error$ErrorType)
long_b_average_error$ErrorType = factor(long_b_average_error$ErrorType)

#plot dataframes for calibrated and uncalibrated euclidean error with 95% confidence interval
ggplot(long_a_average_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))
ggplot(long_b_average_error, aes(x=taskname, y=ErrorValue, fill=ErrorType)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Euclidean Error (degrees)') + scale_fill_discrete(name="Error Type", breaks=c('euclideanError', 'recalibratedEuclideanError'), labels=c('Uncalibrated', 'Recalibrated'))

#Two way ANOVA
#result <- aov(ErrorValue ~ taskname * ErrorType + Error(id/(taskname * ErrorType)), data = long_a_average_error)
#summary(result)
#posthoc <- long_a_average_error %>% group_by(taskname) %>% pairwise_t_test(ErrorValue ~ ErrorType, paired = TRUE, p.adjust.method = "bonferroni")
#posthoc

#result <- aov(ErrorValue ~ taskname * ErrorType + Error(id/(taskname * ErrorType)), data = long_b_average_error)
#summary(result)

long_a_average_error %>% group_by(taskname, ErrorType) %>% shapiro_test(ErrorValue)
m_a_calibration_error <- art(data = long_a_average_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_a_calibration_error)

a_calibration_posthoc <- long_a_average_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
a_calibration_posthoc


long_b_average_error %>% group_by(taskname, ErrorType) %>% shapiro_test(ErrorValue)
m_b_calibration_error <- art(data = long_b_average_error, ErrorValue ~ taskname * ErrorType + Error(id))
anova(m_b_calibration_error)

b_calibration_posthoc <- long_b_average_error %>% group_by(taskname) %>% rstatix::wilcox_test(ErrorValue ~ ErrorType, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
b_calibration_posthoc


#average errors: static recalibrated with calibration task
average_moving_error_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/static_recalibrated_calibration_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')
#average errors: moving recalibrated with calibration task
average_static_error_data <- read.csv("/Users/vivianross/Documents/School/Research/EyeTracking/HoloLensEyeTracking-DataAnalysis-New/Data/moving_recalibrated_calibration_average_error.csv", header = TRUE, fileEncoding = 'UTF-8-BOM')

#add column to specify movement
average_moving_error_data <- average_moving_error_data %>% add_column(movement = 'moving')
average_static_error_data <- average_static_error_data %>% add_column(movement = 'static')

#turn variables into factors
average_moving_error_data$taskname = factor(average_moving_error_data$taskname)
average_moving_error_data$hololens = factor(average_moving_error_data$hololens)
average_moving_error_data$movement = factor(average_moving_error_data$movement)
average_moving_error_data$id = factor(average_moving_error_data$id)
average_static_error_data$taskname = factor(average_static_error_data$taskname)
average_static_error_data$hololens = factor(average_static_error_data$hololens)
average_static_error_data$movement = factor(average_static_error_data$movement)
average_static_error_data$id = factor(average_static_error_data$id)

#remove calibration
average_moving_error_data <- average_moving_error_data[average_moving_error_data$taskname!='calibration',]
average_static_error_data <- average_static_error_data[average_static_error_data$taskname!='calibration',]

#merge dataframes
total_error = rbind(average_moving_error_data, average_static_error_data)

#separate between HoloLenses
a_average_error <- total_error[total_error$hololens=='A',]
b_average_error <- total_error[total_error$hololens=='B',]

#plot cosine error
ggplot(a_average_error, aes(x=taskname, y=cosineError, fill=movement)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Cosine Error (degrees)') + scale_fill_discrete(name="", breaks=c('moving', 'static'), labels=c('Moving', 'Static'))
ggplot(b_average_error, aes(x=taskname, y=cosineError, fill=movement)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Cosine Error (degrees)') + scale_fill_discrete(name="", breaks=c('moving', 'static'), labels=c('Moving', 'Static'))

#Two way ANOVA on cosine error
result <- aov(cosineError ~ taskname * movement + Error(id/(taskname * movement)), data = a_average_error)
summary(result)
posthoc <- a_average_error %>% group_by(taskname) %>% pairwise_t_test(cosineError ~ movement, paired = TRUE, p.adjust.method = "bonferroni")
posthoc

result <- aov(cosineError ~ taskname * movement + Error(id/(taskname * movement)), data = b_average_error)
summary(result)


#plot uncalibrated euclidean error
ggplot(a_average_error, aes(x=taskname, y=euclideanError, fill=movement)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Uncalibrated Euclidean Error (degrees)') + scale_fill_discrete(name="", breaks=c('moving', 'static'), labels=c('Moving', 'Static'))
ggplot(b_average_error, aes(x=taskname, y=euclideanError, fill=movement)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Uncalibrated Euclidean Error (degrees)') + scale_fill_discrete(name="", breaks=c('moving', 'static'), labels=c('Moving', 'Static'))

#Two way ANOVA on uncalibrated euclidean error
#result <- aov(euclideanError ~ taskname * movement + E, data = a_average_error)
#summary(result)
#posthoc <- TukeyHSD(result, which="taskname:movement")
#posthoc

#result <- aov(euclideanError ~ taskname * movement, data = b_average_error)
#summary(result)
#posthoc <- TukeyHSD(result, which="taskname:movement")
#posthoc

a_average_error %>% group_by(taskname, movement) %>% shapiro_test(cosineError)
m_a_movement_error <- art(data = a_average_error,cosineError ~ taskname * movement + Error(id))
anova(m_a_movement_error)

a_movement_posthoc <- a_average_error %>% group_by(taskname) %>% rstatix::wilcox_test(cosineError ~ movement, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
a_movement_posthoc


b_average_error %>% group_by(taskname, movement) %>% shapiro_test(cosineError)
m_b_movement_error <- art(data = b_average_error, cosineError ~ taskname * movement + Error(id))
anova(m_b_movement_error)

b_movement_posthoc <- b_average_error %>% group_by(taskname) %>% rstatix::wilcox_test(cosineError ~ movement, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
b_movement_posthoc


#plot recalibrated euclidean error
ggplot(a_average_error, aes(x=taskname, y=recalibratedEuclideanError, fill=movement)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Recalibrated Euclidean Error (degrees)') + scale_fill_discrete(name="", breaks=c('moving', 'static'), labels=c('Moving', 'Static'))
ggplot(b_average_error, aes(x=taskname, y=recalibratedEuclideanError, fill=movement)) + geom_bar(position=position_dodge(), stat = "summary", fun = "mean") + geom_errorbar(position=position_dodge(), stat="summary", fun.data="mean_se", fun.args = list(mult = 1.96)) + scale_x_discrete(name = "", limits=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), breaks=c('calibration', 'ssHeadConstrained', 'wsBodyConstrained', 'ssWalking', 'wsWalking', 'hallway'), labels=c('Calibration', 'Head Constrained', 'Body Constrained', 'Screen Stabilized Walking', 'World Stabilized Walking', 'Hallway')) + theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) + ylab('Recalibrated Euclidean Error (degrees)') + scale_fill_discrete(name="", breaks=c('moving', 'static'), labels=c('Moving', 'Static'))

#Two way ANOVA on recalibrated euclidean error
result <- aov(recalibratedEuclideanError ~ taskname * movement, data = a_average_error)
summary(result)
posthoc <- TukeyHSD(result, which="taskname:movement")
posthoc

result <- aov(recalibratedEuclideanError ~ taskname * movement, data = b_average_error)
summary(result)
posthoc <- TukeyHSD(result, which="taskname:movement")
posthoc


#Two way ANOVA on cosine error
#result <- aov(cosineError ~ taskname * trial + Error(id/(taskname * trial)), data = a_average_session_error)
#summary(result)
#attach(a_average_session_error)
#posthoc <- pairwise.t.test(cosineError, taskname, p.adj = "bonf", paired = TRUE)
#posthoc
#detach(a_average_session_error)

#result <- aov(cosineError ~ taskname * trial + Error(id/(taskname * trial)), data = b_average_session_error)
#summary(result)

#Two way ART ANOVA on cosine error
#a_average_session_error %>% group_by(taskname, trial) %>% shapiro_test(cosineError) #fails test for normality, can't use normal ANOVA
#m_a_average_session <- art(data = a_average_session_error, cosineError ~ taskname * trial + Error(id))
#anova(m_a_average_session)
#a_average_session_posthoc <- a_average_session_error %>% group_by(taskname) %>% rstatix::wilcox_test(cosineError ~ trial, paired=FALSE, exact=FALSE, p.adjust.method="Bonferroni")
#a_average_session_posthoc

#b_average_session_error %>% group_by(taskname, trial) %>% shapiro_test(cosineError) #fails test for normality, can't use normal ANOVA
#m_b_average_session <- art(data = b_average_session_error, cosineError ~ taskname * trial + Error(id))
#anova(m_b_average_session)
#a_average_session_posthoc <- a_average_session_error %>% group_by(taskname) %>% rstatix::wilcox_test(cosineError ~ trial, paired=FALSE, exact=FALSE, p.adjust.method="Bonferroni")
#a_average_session_posthoc

#trial only
total_session_error %>% group_by(taskname, trial) %>% shapiro_test(cosineError) #fails test for normality, can't use normal ANOVA
m_total_session <- art(data = total_session_error, cosineError ~ taskname * trial + Error(id))
result <- anova(m_total_session)
result

total_session_trial_posthoc <- total_session_error %>% group_by(taskname) %>% rstatix::wilcox_test(cosineError ~ trial, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
total_session_trial_posthoc

total_session_posthoc <- total_session_error %>% rstatix::wilcox_test(cosineError ~ taskname, paired=FALSE, exact=FALSE, p.adjust.method="bonferroni")
total_session_posthoc

DescTools::PostHocTest(result, NULL, "bonferroni", 0.95) #doens't work

