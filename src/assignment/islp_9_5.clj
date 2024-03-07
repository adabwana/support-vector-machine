(ns assignment.islp-9-5
  (:require
    ;plotting
    [scicloj.kindly.v4.kind :as kind]
    ;[aerial.hanami.common :as hc]
    [aerial.hanami.templates :as ht]
    ;[scicloj.metamorph.ml.viz :as ml-viz]
    ;[scicloj.noj.v1.vis.hanami.templates :as vht]
    [scicloj.noj.v1.vis.hanami :as hanami]
    ;maths
    [fastmath.random :as r]
    [fastmath.stats :as stats]
    ;datasets
    [tablecloth.api :as tc]
    [tablecloth.pipeline :as tcm]
    [tech.v3.dataset.metamorph :as dsm]
    [tech.v3.datatype.functional :as dfn]
    ;machine learning
    [scicloj.metamorph.core :as morph]
    [scicloj.metamorph.ml :as ml]
    [scicloj.metamorph.ml.classification :as mlc]
    [scicloj.metamorph.ml.gridsearch :as grid]
    [scicloj.metamorph.ml.loss :as loss]
    [scicloj.ml.smile.classification]
    ;interop
    ;[libpython-clj2.require :refer [require-python]]
    ;[libpython-clj2.python :refer [py. py.. py.-] :as py]
    [scicloj.sklearn-clj.ml]))

;; # ISLP Ch9 Q5
; 5. We have seen that we can ft an SVM with a non-linear kernel in order to perform classification using a non-linear decision boundary. We will now see that we can also obtain a non-linear decision boundary by performing logistic regression using non-linear transformations of the features.
; (a) Generate a data set with n = 500 and p = 2, such that the observations belong to two classes with a quadratic decision boundary between them.
; ## Generate Data
(defn generate-data [len]
  (let [rng (r/rng :isaac 0)
        x1 (map #(- % 0.5)
                (r/->seq (r/distribution :uniform-real {:rng rng}) len))
        x2 (map #(- % 0.5)
                (r/->seq (r/distribution :uniform-real {:rng rng}) len))
        y-int (map #(> (- (Math/pow %1 2) (Math/pow %2 2)) 0) x1 x2)
        y (map {false 0 true 1} y-int)]
    {:x1 x1 :x2 x2 :y y}))

(def data
  (tc/dataset (generate-data 500)))

(tc/head data)
(tc/info data)

; (b) Plot the observations, colored according to their class labels. Your plot should display X1 on the x-axis, and X2 on the y-axis.
; ## Plot data
^kind/vega-lite
(let [plot (tc/rows data :as-maps)]
  {:data     {:values plot}
   :mark     "circle"
   :encoding {:x     {:field :x1 :type "quantitative"}
              :y     {:field :x2 :type "quantitative"}
              :color {:field :y :type "nominal"}}})

(comment
  ;wont plot when rendered
  (-> data
      (hanami/plot ht/point-chart
                   {:X     :x1
                    :Y     :x2
                    :MSIZE 75
                    :COLOR "y"
                    :CTYPE "nominal"})))

;; (c) Fit a logistic regression model to the data, using X1 and X2 as predictors.
; ## Logistic regression
(def response :y)
(def regressors
  (tc/column-names data (complement #{response})))

; ### Model task
; `Pipeline-vanilla` in case want to do more in the ingestion.
(def pipeline-vanilla
  (morph/pipeline
    (dsm/categorical->number [response])
    (dsm/set-inference-target response)))

; Add model context to ingestion.
(defn- create-model-pipeline
  [pipeline-fn model-type params]
  (morph/pipeline
    pipeline-fn
    {:metamorph/id :model}
    (ml/model (merge {:model-type model-type} params))))

; #### Logistic model
(defn logistic-pipe-fn
  [pipeline-fn params]
  (create-model-pipeline pipeline-fn :smile.classification/logistic-regression params))

; ### Evaluate chain
(defn train-test [dataset]
  (tc/split->seq dataset :bootstrap {:seed 123 :repeats 20}))

(defn train-val [dataset]
  (let [ds-split (train-test dataset)]
    (tc/split->seq (:train (first ds-split)) :kfold {:seed 123 :k 5})))

; #### Hyperparameter-grid
(comment
  (ml/hyperparameters :smile.classification/logistic-regression))

(defn generate-hyperparams [model-type]
  (case model-type
    :logistic (take 100
                    (grid/sobol-gridsearch
                      (ml/hyperparameters :smile.classification/logistic-regression)))))

; #### Work-horse
(defn evaluate-pipe [pipe data]
  (ml/evaluate-pipelines
    pipe
    data
    stats/cohens-kappa
    :accuracy
    {:other-metrices                   [{:name      :mathews-cor-coef
                                         :metric-fn stats/mcc}
                                        {:name      :accuracy
                                         :metric-fn loss/classification-accuracy}]
     :return-best-pipeline-only        false
     :return-best-crossvalidation-only true}))

(defn evaluate-model [dataset split-fn model-type model-fn pipeline-fn]
  (let [data-split (split-fn dataset)
        pipelines (map (partial model-fn pipeline-fn) (generate-hyperparams model-type))]
    (evaluate-pipe pipelines data-split)))

; change for different models to test
(def model-type-fns
  {:logistic logistic-pipe-fn})

(defn evaluate-models [dataset split-fn pipeline-fn]
  (mapv (fn [[model-type model-fn]]
          (evaluate-model dataset split-fn model-type model-fn pipeline-fn))
        model-type-fns))

(comment
  ; Alternative: if want to expand `model-type-fns` to simplify `evaluate-models`
  (def model-type-fns
    {:logistic [logistic-pipe-fn pipeline-vanilla]})

  (defn evaluate-model [dataset split-fn model-type model-and-pipeline]
    (let [[model-fn pipeline-fn] model-and-pipeline
          data-split (split-fn dataset)
          pipelines (map (partial model-fn pipeline-fn) (generate-hyperparams model-type))]
      (evaluate-pipe pipelines data-split)))

  (defn evaluate-models [dataset split-fn]
    (mapv (fn [[model-type model-and-pipeline]]
            (evaluate-model dataset split-fn model-type model-and-pipeline))
          model-type-fns))

  (def logistic-models (evaluate-models data train-test)))

; ### View model/s
(defn best-models [eval]
  (->> eval
       flatten
       (map
         #(hash-map :summary (ml/thaw-model (get-in % [:fit-ctx :model]))
                    :fit-ctx (:fit-ctx %)
                    :timing-fit (:timing-fit %)
                    :metric ((comp :metric :test-transform) %)
                    :other-metrices ((comp :other-metrices :test-transform) %)
                    :other-metric-1 ((comp :metric first) ((comp :other-metrices :test-transform) %))
                    :other-metric-2 ((comp :metric second) ((comp :other-metrices :test-transform) %))
                    :params ((comp :options :model :fit-ctx) %)
                    :pipe-fn (:pipe-fn %)))
       (sort-by :metric)))

; ### Get model/s
(comment
  (def logistic-models (evaluate-models data train-test pipeline-vanilla))

  (def logistic-model
    (-> logistic-models
        best-models
        reverse))

  (-> logistic-model first :summary)
  (-> logistic-model first :metric)
  (-> logistic-model first :other-metrices)
  (-> logistic-model first :params)
  ;=>
  ;{:model-type :smile.classification/logistic-regression,
  ; :lambda 79.31055172413794,
  ; :tolerance 1.0E-9,
  ; :max-iterations 9478}
  (-> logistic-model first :fit-ctx :model :options))

(def params
  {:model-type :smile.classification/logistic-regression,
     :lambda 79.31055172413794,
     :tolerance 1.0E-9,
     :max-iterations 9478})


(def logistic-model
  (first (evaluate-pipe
           (map (partial logistic-pipe-fn pipeline-vanilla) params)
           (train-test data))))


; (d) Apply this model to the training data in order to obtain a predicted class label for each training observation. Plot the observations, colored according to the predicted class labels. The decision boundary should be linear.
; ## Visualize model fit
; Function to get the best model's training data.
(defn model->data [model]
  (let [processed-data (-> model first :fit-ctx :model :model-data :smile-df-used)
        keys-vec (-> data
                     (morph/pipe-it (-> model first :pipe-fn))
                     keys)
        without-y (vec (remove #(= :y %) keys-vec))
        with-y-at-end (conj without-y :y)
        data-map (zipmap with-y-at-end processed-data)]
    (tc/dataset data-map)))

(def predictions
  (-> (model->data logistic-model)
      (morph/transform-pipe
        (-> logistic-model first :pipe-fn)
        (-> logistic-model first :fit-ctx))
      :metamorph/data
      :y
      vec))

(def data-predict
  (tc/add-or-replace-column (model->data logistic-model) :y predictions))

^kind/vega-lite
(let [plot (tc/rows data-predict :as-maps)]
  {:data     {:values plot}
   :mark     "circle"
   :encoding {:x     {:field :x1 :type "quantitative"}
              :y     {:field :x2 :type "quantitative"}
              :color {:field :y :type "nominal"}}})


; ### Model performance
; Functions to view performance on full data.
(defn preds
  [model]
  (-> data
      (morph/transform-pipe
        (-> model first :pipe-fn)
        (-> model first :fit-ctx))
      :metamorph/data
      :y
      (->> (map #(long %))
           vec)))

(defn actual
  [model]
  (-> data
      (morph/fit-pipe
        (-> model first :pipe-fn))
      :metamorph/data
      :y
      vec))

(defn evaluate-predictions
  "Evaluates predictions against actual labels, returns confusion map and metrics."
  [preds actual]
  (let [conf-map (mlc/confusion-map->ds (mlc/confusion-map preds actual :none))
        kappa (stats/cohens-kappa preds actual)
        mcc (stats/mcc preds actual)]
    {:confusion-map conf-map
     :cohens-kappa  kappa
     :mcc           mcc}))

^kind/dataset
(evaluate-predictions (preds logistic-model) (actual logistic-model))

;; (e) Now fit a logistic regression model to the data using non-linear  functions of X$_1$ and X$_2$ as predictors (e.g. $X_1^2$, $X_1*X_2$, $log(X_2)$, and so forth)
; ## Non-linear regressors
; Making pipelines (like `vanilla`) to apply to `evaluate-models`
(def pipeline-squared
  (morph/pipeline
    (tcm/add-or-replace-columns {:x1-sq (fn [row]
                                          (map #(Math/pow % 2) (:x1 row)))
                                 :x2-sq (fn [row]
                                          (map #(Math/pow % 2) (:x2 row)))})
    (dsm/categorical->number [response])
    (dsm/set-inference-target response)))

(def pipeline-interact
  (morph/pipeline
    (tcm/add-or-replace-columns {:x1-x2 (fn [ds] (dfn/* (:x1 ds)
                                                        (:x2 ds)))})
    (dsm/categorical->number [response])
    (dsm/set-inference-target response)))

(def pipeline-combined
  (morph/pipeline
    (tcm/add-or-replace-columns {:x1-x2 (fn [ds] (dfn/* (:x1 ds)
                                                        (:x2 ds)))
                                 :x1-sq (fn [row]
                                          (map #(Math/pow % 2) (:x1 row)))
                                 :x2-sq (fn [row]
                                          (map #(Math/pow % 2) (:x2 row)))})
    (dsm/categorical->number [response])
    (dsm/set-inference-target response)))

; ### Set the model `case` name and respective model pipeline
(def model-type-fns
  {:logistic logistic-pipe-fn})

(comment
  (def pipe-squared
    (evaluate-models data train-test pipeline-squared))
  (def pipe-interact
    (evaluate-models data train-test pipeline-interact))
  (def pipe-combined
  (evaluate-models data train-test pipeline-combined)))

; Don't want to run the full evaluate models
; Squared
(def params
  {:model-type :smile.classification/logistic-regression,
   :lambda 0.001,
   :tolerance 0.07894736863157896,
   :max-iterations 3747})

(def pipe-squared
  (first (evaluate-pipe
           (map (partial logistic-pipe-fn pipeline-squared) params)
           (train-test data))))

(def logistic-squared
  (-> pipe-squared best-models reverse))
(-> logistic-squared first :fit-ctx :model :feature-columns)

; Interact
(def params
  {:model-type :smile.classification/logistic-regression,
   :lambda 41.37989655172413,
   :tolerance 1.0E-9,
   :max-iterations 4268})

(def pipe-interact
  (first (evaluate-pipe
           (map (partial logistic-pipe-fn pipeline-interact) params)
           (train-test data))))

(def logistic-interact
  (-> pipe-interact best-models reverse))
(-> logistic-interact first :fit-ctx :model :feature-columns)

; Combined
(def params
  {:model-type :smile.classification/logistic-regression,
   :lambda 0.001,
   :tolerance 0.07894736863157896,
   :max-iterations 3747}  )

(def pipe-combined
  (first (evaluate-pipe
           (map (partial logistic-pipe-fn pipeline-combined) params)
           (train-test data))))

(def logistic-combined
  (-> pipe-combined best-models reverse))
(-> logistic-combined first :fit-ctx :model :feature-columns)

^kind/dataset
(evaluate-predictions (preds logistic-squared) (actual logistic-squared))
^kind/dataset
(evaluate-predictions (preds logistic-interact) (actual logistic-interact))
^kind/dataset
(evaluate-predictions (preds logistic-combined) (actual logistic-combined))

;(f) Apply this model to the training data in order to obtain a predicted class label for each training observation. Plot the observations, colored according to the predicted class labels. The decision boundary should be obviously non-linear. If it is not, then repeat (a)–(e) until you come up with an example in which the predicted class labels are obviously non-linear.
; ## Non-linear logistic visualization
(def data-plot
  (tc/add-or-replace-column data :y (preds logistic-interact)))

^kind/vega-lite
(let [plot (tc/rows data-plot :as-maps)]
  {:data     {:values plot}
   :mark     "circle"
   :encoding {:x     {:field :x1 :type "quantitative"}
              :y     {:field :x2 :type "quantitative"}
              :color {:field :y :type "nominal"}}})

(def data-plot
  (tc/add-or-replace-column data :y (preds logistic-squared)))

^kind/vega-lite
(let [plot (tc/rows data-plot :as-maps)]
  {:data     {:values plot}
   :mark     "circle"
   :encoding {:x     {:field :x1 :type "quantitative"}
              :y     {:field :x2 :type "quantitative"}
              :color {:field :y :type "nominal"}}})

;(g) Fit a support vector classifer to the data with X1 and X2 as predictors. Obtain a class prediction for each training observation. Plot the observations, colored according to the predicted class labels.
; ## Support vector visualization
; Many issues with sklearn. One of which, major parameter "C" cost, does not work.
(comment
  ;failed a lot, checked and found "c" cant be param, strange
  (def svm-pipe
    (morph/pipeline
      (dsm/categorical->number [response])
      (dsm/set-inference-target response)
      {:metamorph/id :model}
      (ml/model {:model-type     :sklearn.classification/svc
                 :kernel         "poly"
                 :degree         7
                 ;:c 0.001 ;doesnt work with c???
                 :predict-proba? false})))

  (def ds-split
    (tc/split->seq data :bootstrap {:seed 123 :repeats 20}))

  (evaluate-pipe [svm-pipe] ds-split))

; ### SVM context pipeline
; Like the logistic pipeline context.
(defn svm-pipe-fn
  [pipeline-fn params]
  (create-model-pipeline pipeline-fn :sklearn.classification/svc params))

(comment ;nil
  (ml/hyperparameters :sklearn.classification/svc))

; ### Set the model `case` name and respective model pipeline
(defn generate-hyperparams [model-type]
  (case model-type
    :svm-linear (grid/sobol-gridsearch
                  {:kernel         "linear"
                   :predict-proba? false})
    :svm-rbf (grid/sobol-gridsearch
               {:predict-proba? false})
    :svm-sigmoid (grid/sobol-gridsearch
                   {:kernel         "sigmoid"
                    :predict-proba? false})
    :svm-poly (grid/sobol-gridsearch
                {:kernel         "poly"
                 :degree         (grid/linear 1 4 4 :int32)
                 :predict-proba? false})))

(def model-type-fns
  {:svm-linear  svm-pipe-fn
   :svm-rbf     svm-pipe-fn
   :svm-sigmoid svm-pipe-fn
   :svm-poly    svm-pipe-fn})

; ### Collect all four SVM models.
(def svm-models (evaluate-models data train-test pipeline-vanilla))

; ### Extract first for the `:svm-linear.
(def svm-linear
  (-> svm-models first best-models reverse))
(-> svm-linear first :fit-ctx :model :options)

^kind/dataset
(evaluate-predictions (preds svm-linear) (actual svm-linear))

(def data-plot
  (tc/add-or-replace-column data :y (preds svm-linear)))

^kind/vega-lite
(let [plot (tc/rows data-plot :as-maps)]
  {:data     {:values plot}
   :mark     "circle"
   :encoding {:x     {:field :x1 :type "quantitative"}
              :y     {:field :x2 :type "quantitative"}
              :color {:field :y :type "nominal"}}})

; (h) Fit a SVM using a non-linear kernel to the data. Obtain a class prediction for each training observation. Plot the observations,colored according to the predicted class labels
; ## Non-linear kernels
; Already made in the last question, now extract from the `svm-models` collection.
(def svm-rbf
  (-> svm-models second best-models reverse))
(-> svm-rbf first :fit-ctx :model :options)

(def data-plot
  (tc/add-or-replace-column data :y (preds svm-rbf)))

^kind/vega-lite
(let [plot (tc/rows data-plot :as-maps)]
  {:data     {:values plot}
   :mark     "circle"
   :encoding {:x     {:field :x1 :type "quantitative"}
              :y     {:field :x2 :type "quantitative"}
              :color {:field :y :type "nominal"}}})

^kind/dataset
(evaluate-predictions (preds svm-rbf) (actual svm-rbf))

(def svm-sigmoid
  (-> svm-models rest second best-models reverse))
(-> svm-sigmoid first :fit-ctx :model :options)

(def data-plot
  (tc/add-or-replace-column data :y (preds svm-sigmoid)))

^kind/vega-lite
(let [plot (tc/rows data-plot :as-maps)]
  {:data     {:values plot}
   :mark     "circle"
   :encoding {:x     {:field :x1 :type "quantitative"}
              :y     {:field :x2 :type "quantitative"}
              :color {:field :y :type "nominal"}}})

^kind/dataset
(evaluate-predictions (preds svm-sigmoid) (actual svm-sigmoid))


(def svm-poly
  (-> svm-models last best-models reverse))
(-> svm-poly first :fit-ctx :model :options)

(def data-plot
  (tc/add-or-replace-column data :y (preds svm-poly)))

^kind/vega-lite
(let [plot (tc/rows data-plot :as-maps)]
  {:data     {:values plot}
   :mark     "circle"
   :encoding {:x     {:field :x1 :type "quantitative"}
              :y     {:field :x2 :type "quantitative"}
              :color {:field :y :type "nominal"}}})

^kind/dataset
(evaluate-predictions (preds svm-poly) (actual svm-poly))

;(i) Comment on your results.
; `Linear` and `Sigmoid` kernels in our SVM models perform poorly on certain polynomial relationships. We can model those relationships "linearly" by creating polynomial regressors in our model. Another way is to use different SVM kernel functions. `Scikit-learn`'s default kernel function--`RBF`--provided a nice non-linear option. Also, the `poly` kernel was able to identify that degree two is the appropriate power for our data. `RBF` and `poly` performed similarly on our data, while `poly` is the best choice with a Kappa score of 0.98.