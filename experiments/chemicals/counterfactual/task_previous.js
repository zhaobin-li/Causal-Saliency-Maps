const simulate = false;
const debugMode = mode === "debug"; // psiturk/templates/default.html:29

const psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);

const numDemo = 5;
const numDemoRange = [...Array(numDemo).keys()];

const numTrials = debugMode ? 10 : 30;
const numTrialsRange = [...Array(numTrials).keys()].map((x) => x + numDemo);

const salPrompt = `The brightly-colored salient molecules are important to ${getInlineIcon(
  "robot1"
)}'s decision. ${getInlineIcon(
  "robot1"
)} thinks the non-salient molecules don't match its label or don't know which label to assign to those molecules`;

const jsPsych = initJsPsych({
  show_progress_bar: true,
  auto_update_progress_bar: false,

  on_trial_finish: function (data) {
    if (debugMode) {
      console.log(jsPsych.data.get().json(true));
    }
  },

  on_finish: function (data) {
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

numDemoRange.forEach(
  (i) =>
    (images = images.concat(
      ["mol", "sal", "opp", "same"].map((molImg) => getMolPath(molImg, i))
    ))
);

// main trials don't require molecules image
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

const gender = {
  type: jsPsychSurveyMultiChoice,
  questions: [
    {
      prompt: "What is your gender? (optional)",
      options: ["Male", "Female", "Other"],
      required: false,
      name: "gender",
    },
  ],
};

const age = {
  type: jsPsychSurveyText,
  questions: [
    {
      prompt: "What is your age? (optional)",
      required: false,
      name: "age",
    },
  ],
};

// timeline.push(gender);
// timeline.push(age);

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
    )} will also give us a "saliency map", which tells us where it is looking at to give its label.</p> 
     <p class="important"><strong>Important:</strong> ${salPrompt}. </p>
     <img src="/static/images/colorbar.svg" class="colorbar">
     <div class="grid-col">
     <div class="robot-label-img">${getRobotwLabel(
       "robot1",
       "toxic"
     )}<img src=${getMolPath("sal", 0)}></div>
     <div class="robot-label-img">${getRobotwLabel(
       "robot1",
       "safe"
     )}<img src=${getMolPath("sal", 1)}></div>
     </div>`,

    `<p><p class="important"><strong>Task: </strong>Given two manipulated images with some molecules made invisible, click on the one which will cause ${getInlineIcon(
      "robot1"
    )}'s label to change or remain the same. Let's practice!</p>`,
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
            jsPsych.timelineVariable("saliencyLabel")
          )} and tells us where it is looking at.`;

          return `<p>${originalStr}</p><div class="robot-label-img">${getRobotwLabel(
            "robot1",
            jsPsych.timelineVariable("saliencyLabel")
          )}<img src=${jsPsych.timelineVariable(
            "sal"
          )} ></div><div class="prompt">${getRobotwLabel(
            "robot1",
            jsPsych.timelineVariable("promptLabel")
          )}</div>`;
        },
        prompt: () =>
          `<p>Which manipulated image will cause ${getInlineIcon(
            "robot1"
          )} to label the occluded image as ${getInlineIcon(
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
            data.saliencyLabel === data.promptLabel ? "same" : "opp";

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

const timelineFactory = (saliencyLabel, promptLabel, imgNum, buttonOrder) => {
  return {
    saliencyLabel,
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
      timelineFactory("toxic", "safe", 3, ["same", "opp"]),
    ]),

    {
      type: jsPsychInstructions,
      pages: [
        `<p class="important"><strong>Reminder:</strong> ${salPrompt}. Let's begin! </p>`,
      ],
      show_clickable_nav: true,
    },
  ],
  on_timeline_finish: () => jsPsych.setProgressBar(0.2),
};
timeline.push(practice);

const mainTimelineVariables = [];

numTrialsRange.forEach((numImg) =>
  mainTimelineVariables.push(
    timelineFactory(
      sampleOne(["safe", "toxic"]),
      sampleOne(["safe", "toxic"]),
      numImg,
      jsPsych.randomization.shuffle(["same", "opp"])
    )
  )
);

timeline.push(mainFactory(mainTimelineVariables, false));

const feedback = {
  type: jsPsychSurveyText,
  questions: [
    {
      prompt: "Any feedback? (optional)",
      required: false,
      name: "feedback",
    },
  ],
};

timeline.push(feedback);

if (debugMode) {
  timeline.push({
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function () {
      const mainTrials = jsPsych.data.get().filter({ trial: "results" });

      console.assert(mainTrials.count() === numTrials);

      return `<p class="important"><strong>Debugging: </strong>Press any key to complete experiment. Thank you!</p>
        
      <p>Average RT: ${Math.round(mainTrials.select("rt").mean())}ms</p>`;
    },
  });
}

if (simulate) {
  jsPsych.simulate(timeline, "data-only");
} else {
  jsPsych.run(timeline);
}
