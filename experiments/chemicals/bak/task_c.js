// psiturk/templates/default.html:29 activated by visiting app url directly
// can leave in production mode because won't be activated by mturkers
const debugMode = mode === "debug";
const simulate = false;

const psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);

const numDemo = 10;
const numDemoRange = [...Array(numDemo).keys()];

const numTrials = debugMode ? 30 : 30;
const numTrialsRange = [...Array(numTrials).keys()].map((x) => x + numDemo);

const salPrompt = `The brightly-colored salient molecules are <strong>important</strong> to ${getInlineIcon(
  "robot1"
)}'s decision. ${getInlineIcon(
  "robot1"
)} thinks the non-salient molecules don't match its label or don't know which label to assign to those molecules.`;

const aimPrompt = `Click on the occluded image which is <strong>consistent</strong> with ${getInlineIcon(
  "robot1"
)}'s new label.`;

const jsPsych = initJsPsych({
  show_progress_bar: true,
  auto_update_progress_bar: false,

  on_trial_finish: function (data) {
    if (debugMode) {
      console.log(jsPsych.data.get().last(1).json(true));
    }
  },

  on_finish: function (data) {
    const mainTrials = jsPsych.data.get().filter({ trial: "results" });
    psiTurk.recordUnstructuredData(
      "meanMainRTsecs",
      mainTrials.select("rt").mean() / 1000
    );

    const finalTrial = jsPsych.data.get().last(1);
    psiTurk.recordUnstructuredData(
      "totalTimeElapsedmins",
      finalTrial.select("time_elapsed").values[0] / 1000 / 60
    );

    psiTurk.recordTrialData(data);
    psiTurk.saveData({
      success: () => psiTurk.completeHIT(),
    });

    if (debugMode) {
      jsPsych.data.get().localSave("csv", "debug.csv");
    }
  },
});

const timeline = [];

const minScreenResolution = {
  type: jsPsychBrowserCheck,
  minimum_width: 800,
  minimum_height: 600,
};

const browserCheck = {
  type: jsPsychBrowserCheck,
  inclusion_function: (data) => {
    return ["chrome"].includes(data.browser);
  },
  exclusion_message: () =>
    `<p>You must use Chrome to complete this experiment.</p>`,
};

timeline.push(minScreenResolution);
timeline.push(browserCheck);

let images = [];

// only demo require molecules image
numDemoRange.forEach(
  (i) =>
    (images = images.concat(
      ["mol", "sal", "opp", "same"].map((molImg) => getMolPath(molImg, i))
    ))
);

numTrialsRange.forEach(
  (i) =>
    (images = images.concat(
      ["sal", "opp", "same"].map((molImg) => getMolPath(molImg, i))
    ))
);

images.push("/static/images/colorbar.svg");

const preload = {
  type: jsPsychPreload,
  auto_preload: true,
  images: images,
};

timeline.push(preload);

const instructions = {
  type: jsPsychInstructions,
  pages: [
    `<p>Glorbian aliens are exploring newly discovered chemicals in a parallel universe and are testing robot ${getInlineIcon(
      "robot1"
    )} to label these chemicals as ${getInlineIcon("safe")} or ${getInlineIcon(
      "toxic"
    )}. </p><div class="grid-col">${getBlockIcon(
      "robot1",
      "fa-shake"
    )}</div><div class="grid-col">${getBlockIcon(
      "safe",
      "fa-fade"
    )}${getBlockIcon("toxic", "fa-fade")}</div>`,

    `<p>For instance, ${getInlineIcon(
      "robot1"
    )} labels the left image as ${getInlineIcon(
      "toxic"
    )} and the right image as ${getInlineIcon("safe")}.</p>

     <div class="grid-col">
     <div class="robot-label-img">${getRobotwLabel(
       "robot1",
       "toxic"
     )}<img src=${getMolPath("mol", 0)}></div>
     <div class="robot-label-img">${getRobotwLabel(
       "robot1",
       "safe"
     )}<img src=${getMolPath("mol", 1)}></div>
     </div>`,

    `<p>${getInlineIcon(
      "robot1"
    )} also gives us its "saliency map", telling us where it is looking at to make its decision.</p> <p class="important"><strong>Important:</strong> ${salPrompt} </p><img src="/static/images/colorbar.svg" class="colorbar">
     <div class="grid-col">
     <div class="robot-label-img">${getRobotwLabel(
       "robot1",
       "safe"
     )}<img src=${getMolPath("sal", 0)}></div>
     <div class="robot-label-img">${getRobotwLabel(
       "robot1",
       "toxic"
     )}<img src=${getMolPath("sal", 1)}></div>
     </div>`,

    `<p><p class="important"><strong>Task: </strong>On every trial, you are given two occluded images in which some molecules are made invisible. ${getInlineIcon(
      "robot1"
    )} will also give a new label, which may or may not be the same as its original label over the saliency image. </p><p class="important"><strong>Aim: </strong>${aimPrompt} Let's warm up!</p>`,
  ],
  show_clickable_nav: true,
  on_finish: () => jsPsych.setProgressBar(0.1),
};

timeline.push(instructions);

const mainFactory = (timelineVariables, isPractice = true) => {
  return {
    timeline: [
      {
        type: jsPsychHtmlButtonResponse,
        stimulus: () => {
          const originalStr = `${getInlineIcon(
            "robot1"
          )} thinks the image is ${getInlineIcon(
            jsPsych.timelineVariable("salLabel")
          )} and tells us where it is looking at.`;

          return `<p>${originalStr}</p><div class="robot-label-img">${getRobotwLabel(
            "robot1",
            jsPsych.timelineVariable("salLabel")
          )}<img src=${jsPsych.timelineVariable(
            "sal"
          )}></div><div class="prompt">${getRobotwLabel(
            "robot1",
            jsPsych.timelineVariable("promptLabel")
          )}</div>`;
        },
        prompt: () =>
          `<p>Which occluded image will ${getInlineIcon(
            "robot1"
          )} label as ${getInlineIcon(
            jsPsych.timelineVariable("promptLabel")
          )}?</p>`,
        choices: jsPsych.timelineVariable("buttonPaths"),
        button_html: `<img src=%choice%>`,
        data: {
          trial: isPractice ? "practice" : "results",
        },
        on_finish: (data) => {
          Object.assign(data, jsPsych.getAllTimelineVariables());

          data.responseCompare = data.buttonOrder[data.response];

          data.labelCompare =
            data.salLabel === data.promptLabel ? "same" : "opp";

          if (!isPractice) {
            jsPsych.setProgressBar(
              jsPsych.getProgressBarCompleted() + (1 / numTrials) * 0.7 // 1 - instructions - practice
            );
          }
        },
      },
    ],
    timeline_variables: timelineVariables,
    randomize_order: !isPractice,
  };
};

const timelineFactory = (salLabel, promptLabel, imgNum, buttonOrder) => {
  return {
    salLabel,
    promptLabel,

    imgNum,
    buttonOrder,

    get sal() {
      return getMolPath("sal", this.imgNum);
    },

    get buttonPaths() {
      return this.buttonOrder.map((molImg) => getMolPath(molImg, this.imgNum));
    },
  };
};

const practice = {
  timeline: [
    mainFactory([
      timelineFactory("safe", "safe", 2, ["opp", "same"]),
      timelineFactory("safe", "toxic", 3, ["same", "opp"]),
      timelineFactory("toxic", "safe", 4, ["opp", "same"]),
      timelineFactory("toxic", "toxic", 5, ["same", "opp"]),
    ]),
    {
      type: jsPsychInstructions,
      pages: [
        `<p class="important"><strong>Reminder:</strong> ${salPrompt} ${aimPrompt} Let's begin! </p>`,
      ],
      show_clickable_nav: true,
    },
  ],
  on_timeline_finish: () => jsPsych.setProgressBar(0.2),
};

timeline.push(practice);

const mainTimelineVariables = [];

numTrialsRange.forEach((imgNum) =>
  mainTimelineVariables.push(
    timelineFactory(
      sampleOne(["safe", "toxic"]),
      sampleOne(["safe", "toxic"]),
      imgNum,
      jsPsych.randomization.shuffle(["same", "opp"])
    )
  )
);

timeline.push(mainFactory(mainTimelineVariables, false));

// required or else nobody will answer
const gender = {
  type: jsPsychSurveyMultiChoice,
  questions: [
    {
      prompt: "What is your gender?",
      options: ["Male", "Female", "Other"],
      required: true,
      name: "gender",
    },
  ],
  on_finish: (data) =>
    psiTurk.recordUnstructuredData("gender", data.response["gender"]),
};

const age = {
  type: jsPsychSurveyText,
  questions: [
    {
      prompt: "What is your age?",
      required: true,
      name: "age",
    },
  ],
  on_finish: (data) =>
    psiTurk.recordUnstructuredData("age", data.response["age"]),
};

const feedback = {
  type: jsPsychSurveyText,
  questions: [
    {
      prompt: "Any feedback? (optional)",
      required: false,
      name: "feedback",
    },
  ],
  on_finish: (data) =>
    psiTurk.recordUnstructuredData("feedback", data.response["feedback"]),
};

timeline.push(gender);
timeline.push(age);
timeline.push(feedback);

if (simulate) {
  jsPsych.simulate(timeline, "data-only");
} else {
  jsPsych.run(timeline);
}
