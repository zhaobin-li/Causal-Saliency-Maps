library(jsonlite)

library(tidyverse)
library(ggplot2)

library(lme4)
library(afex)


MIN_AVG_RT <- 1000 # ms
NUM_TRIALS <- 30

# Wrangle ------------------------------------------------------------


# column names in https://psiturk.readthedocs.io/en/stable/command_line.html?highlight=trialdata.csv#download-datafiles
df <-
  read_csv("trialdata_3robots.csv",
    col_names = c("id", "trialNum", "time", "trialData")
  ) |>
  arrange(time) |> # arrange by time
  glimpse()

# analyse 1 participant
# df <- df |> filter(id %in% c("debugQfD0W:debug1rA1P")) |> glimpse()

# exclude by time
df <- df |>
  filter(time >= 1656597315813) |>
  glimpse()

# remove debug attempts
# df <- df |>
#   filter(!str_detect(id, "debug")) |>
#   glimpse()

# # anonymize id
# df <- df |>
#   group_by(id) |>
#   mutate(id = cur_group_id()) |>
#   glimpse()

# check no duplicate ids
df |>
  group_by(id) |>
  filter(n() > 1)


# get nested JSON trial data
df <-
  df |>
  rowwise() |>
  mutate(trialData = fromJSON(trialData)) |>
  unnest(trialData) |>
  glimpse()

df |> write_csv("wrangledata.csv")

# get time to complete experiment
df |>
  group_by(id) |>
  select(time_elapsed) |>
  filter(row_number() == n()) |>
  mutate(time_elapsed_min = time_elapsed / 1000 / 60) |> # time_elapsed in ms
  arrange(desc(time_elapsed_min))

# restrict to main trials
df <- df |>
  filter(trial == "results") |>
  glimpse()

# # restrict to practice trials
# df <- df |>
#   filter(trial == "practice") |>
#   glimpse()

# get rt
df |>
  group_by(id) |>
  summarize(mean_rt = mean(rt)) |>
  arrange(mean_rt)

# participants with rt < MIN_AVG_RT
df |>
  group_by(id) |>
  summarize(mean_rt = mean(rt)) |>
  filter(mean_rt < MIN_AVG_RT)

# exclude rt < MIN_AVG_RT
# df |>
#   group_by(id) |>
#   filter(mean(rt) > MIN_AVG_RT)


# Visualize ---------------------------------------------------------------

df <-
  df |>
  select(id, imgNum, responseCompare, robotCondition, labelCompare) |>
  glimpse()

df <- df |>
  mutate(
    responseCompare = factor(responseCompare),
    robotCondition = factor(robotCondition),
    labelCompare = factor(labelCompare),
  ) |>
  glimpse()


df |>
  group_by(robotCondition, labelCompare) |>
  count(id, responseCompare) |>
  complete(id, responseCompare, fill = list(n = 0)) |>
  ggplot(aes(responseCompare, n / 30)) +
  geom_boxplot() +
  geom_jitter() +
  facet_grid(vars(labelCompare), vars(robotCondition), labeller = label_both) +
  labs(y = "P(responseCompare)")

df |>
  group_by(robot) |>
  count(id, choice, .drop = FALSE) |>
  ggplot(aes(robot, n / 30, color = choice)) +
  geom_boxplot() +
  geom_jitter()

df |>
  group_by(label) |>
  count(id, choice, .drop = FALSE) |>
  ggplot(aes(label, n / 30, color = choice)) +
  geom_boxplot() +
  geom_point(position = position_jitterdodge()) +
  labs(y = "P(choice)")

df |>
  group_by(labelCompare) |>
  count(id, responseCompare, .drop = FALSE) |>
  ggplot(aes(labelCompare, n / 30, color = responseCompare)) +
  geom_boxplot() +
  geom_point(position = position_jitterdodge()) +
  labs(y = "P(choice)")



# Analyze ----------------------------------------------------------------

df <- df |>
  mutate(
    choice = factor(ifelse(responseCompare == "same", 0, 1)),
    robot = relevel(factor(robotCondition), ref = "same"),
    label = factor(ifelse(labelCompare == "same", 0, 1)),
  ) |>
  glimpse()


# no need to add interaction with deterministic robots
model.glm <- glm(choice ~ robot + label,
  data = df,
  family = binomial(link = "logit")
)
summary(model.glm)

# intercept only since robot between id
# tried adding by-id and by-imgNum label but don't converge
model.glmer <-
  glmer(choice ~ robot + label + (1 | id) + (1 | imgNum),
    data = df,
    family = binomial
  )

summary(model.glmer)

mixed(model.glmer,
  data = df,
  family = binomial, method = "LRT"
)


# sameRobot only ----------------------------------------------------------

dfOne <- df |> filter(robot == "random")


# no need to add interaction with deterministic robots
model.glm <- glm(choice ~ 1,
  data = dfOne,
  family = binomial(link = "logit")
)
summary(model.glm)

# intercept only since robot between id
# tried adding by-id and by-imgNum label but don't converge
model.glmer <-
  glmer(choice ~ 1 + (1 | id),
    data = dfOne,
    family = binomial
  )

summary(model.glmer)


# Simulate H1 Label Only -------------------------------------------------------------


choiceEqualLabelPr <- 3 / 4

df <- df |>
  rowwise() |>
  mutate(choiceH1LabelOnly = sample(
    c(label, 1 - label),
    1,
    prob = c(
      choiceEqualLabelPr,
      1 - choiceEqualLabelPr
    )
  )) |>
  ungroup()

df |>
  summarize(mean(choiceH1LabelOnly == label))

df |>
  group_by(id, robot, label, choiceH1LabelOnly) |>
  tally() |>
  ggplot(aes(factor(choiceH1LabelOnly), n)) +
  geom_point() +
  geom_boxplot() +
  facet_grid(cols = vars(robot, label), labeller = label_context)

model.H1Label <- glm(choiceH1LabelOnly ~ robot * label,
  data = df,
  family = binomial(link = "logit")
)
summary(model.H1Label)

model.glmer <-
  glmer(choiceH1LabelOnly ~ label * robot + (1 | id),
    data = df,
    family = binomial()
  )

summary(model.glmer)


# Simulate H1 Label + Robot -------------------------------------------------------------


choiceEqualLabelPr <- 3 / 4

df <- df |>
  rowwise() |>
  mutate(choiceH1LabelRobot = case_when(
    robot %in% c("sameRobot", "sameLabel") ~ sample(
      c(label, 1 - label),
      1,
      prob = c(choiceEqualLabelPr, 1 - choiceEqualLabelPr)
    ),
    TRUE ~ choice
  )) |>
  ungroup()

df |>
  group_by(robot) |>
  summarize(mean(choiceH1LabelRobot == label))

df |>
  group_by(id, robot, label, choiceH1LabelRobot) |>
  tally() |>
  ggplot(aes(factor(choiceH1LabelRobot), n)) +
  geom_point() +
  geom_boxplot() +
  facet_grid(cols = vars(robot, label), labeller = label_context)

model.H1LabelRobot <- glm(
  choiceH1LabelRobot ~ robot * label,
  data =  df,
  family = binomial(link = "logit")
)
summary(model.H1LabelRobot)


model <-
  glmer(choice ~ 1 + label + robot + (1 | id) + (1 | imgNum),
    data = df,
    family = binomial
  )

model <-
  glmer(choiceH1LabelRobot ~ label * robot + (1 | id),
    data = df,
    family = binomial
  )
