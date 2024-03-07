(ns assignment.islp-9-7
  (:require
    [calc-metric.patch]
    [clojisr.v1.applications.plotting
     :refer [plot->svg]]
    [clojisr.v1.r :refer [bra r+ r- r->clj clj->r]]
    [clojisr.v1.require :refer [require-r]]
    [fastmath.stats :as stats]
    [scicloj.kindly.v4.kind :as kind]
    [scicloj.metamorph.core :as morph]
    [scicloj.metamorph.ml.gridsearch :as grid]
    [tablecloth.api :as tc]))

(comment
  (clojure.java.shell/sh "which" "R"))

;; # ISLP Ch9 Q7
; In this problem, you will use support vector approaches in order to predict whether a given car gets high or low gas mileage based on the Auto data set.
; (a) Create a binary variable that takes on a 1 for cars with gas mileage above the median, and a 0 for cars with gas mileage below the median.
; ## Binary response
; Load the required R libraries
(require-r '[base :refer [RNGkind set-seed summary plot $ expand-grid which-max subset
                          as-numeric factor levels as-character data-frame append]]
           '[stats :refer [predict]]
           '[ISLR :as islr]
           '[caret :refer [createDataPartition trainControl modelLookup train
                           defaultSummary prSummary twoClassSummary mnLogLoss]]
           '[kernlab]
           '[e1071]
           '[ggplot2 :refer [ggplot aes geom_point geom_line
                             facet_wrap theme_bw]])

;; Call in datasets from R. This one comes from the `islr` library in a dataset called `Auto`.
(def auto
  (-> (r->clj islr/Auto)
      (tc/drop-columns :$row.names)))

(stats/median (:mpg auto))

;; ### Create binary response
(def auto-cat
  (-> auto
      (tc/map-columns :mpg-cat [:mpg]
                      #(if (>= % (stats/median (:mpg auto))) 1 0))
      (tc/map-columns :mpg-cat str)))

; Later, working with the data, R doesn't like `:keywords` like Clojure does. Create a R-compatible data.frame. Notice, I'm flipping through both Clojure and R data structures in R functions and vice cersa.
(def r-data
  (tc/rename-columns auto-cat (fn [col]
                                (-> col
                                    name
                                    (clojure.string/replace #"-" ".")))))

(keys auto-cat)
(keys r-data)

(summary auto-cat)

;; (b) Fit a support vector classifier to the data with various values of C, in order to predict whether a car gets high or low gas mileage. Report the cross-validation errors associated with different values of this parameter. Comment on your results. Note you will need to fit the classifier without the gas mileage variable to produce sensible results
;; ## Fit SVMs
;; Partition data
(def index
  (createDataPartition :y ($ r-data 'mpg.cat)
                       :p 0.7 :list false))

; Train and test data
(def training-data
  (bra r-data index nil))
(def test-data
  (bra r-data (r- index) nil))

; Caret svmLinear
(RNGkind :sample.kind "Rounding")
(set-seed 0)

;; Bootstrap cross-validation
(def train-control
  (trainControl :method "boot" :number 20))

(modelLookup "svmLinear")

;; Build model
(def svm-linear
  (train '(tilde mpg.cat (- . mpg))
         :data training-data :method "svmLinear"
         :trControl train-control :metric "kappa"
         :tuneGrid (expand-grid :C (range 0.01 0.125 0.025))))

;; View final model
($ svm-linear 'finalModel)

(comment
  (def plot-svm-linear
    (r.e1071/svm '(formula mpg.cat (- . mpg))
                 :data (tc/convert-types r-data "mpg.cat" :int32)
                 :kernel "linear" :cost 0.01)))

;; ### Cross-validation errors
($ svm-linear 'results)

; I measured the goodness-of-fit versus errors. For each C, R built 20 bootstraped linear SVM models. The best Kappa per C is reported.  Based on Kappa, $C = 0.035$ is best.

;; (c) Now repeat (b), this time using SVMs with radial and polynomial basis kernels, with different values of gamma and degree and C. Comment on your results.
; ## SVM radial and polynomial
; ### svmRadial
(RNGkind :sample.kind "Rounding")
(set-seed 0)

; Hyperparameter check
(modelLookup "svmRadial")

;; Build model
(def svm-radial
  (train '(formula mpg.cat (- . mpg))
         :data training-data :method "svmRadial" :trControl train-control
         :metric "kappa"
         :tuneGrid (tc/dataset
                     (grid/sobol-gridsearch
                       {:C     (grid/linear 0.25 150 8)
                        :sigma (grid/linear 0.000001 0.00001 5)}))))

;; View final model
($ svm-radial 'finalModel)

(comment
  (def plot-svm-radial
    (r.e1071/svm '(formula mpg.cat (- . mpg))
                 :data (tc/convert-types r-data "mpg.cat" :int32)
                 :kernel "radial" :cost 107.214285714286 :sigma 3.25e-06)))

;; ### Cross-validation errors
($ svm-radial 'results)

; Much bigger grid to search through. A plot would make this easier.

; ### svmRadial
(RNGkind :sample.kind "Rounding")
(set-seed 0)

; Hyperparameter check
(modelLookup "svmPoly")

;; Build model
(comment
  ;too much time
  (def svm-poly
    (train '(tilde mpg.cat
                   (+ cylinders displacement horsepower weight
                      acceleration year origin name))
           :data training-data :method "svmPoly"
           :trControl train-control :metric "kappa"
           :tuneGrid (tc/dataset
                       (take 6
                             (grid/sobol-gridsearch
                               {:C      (grid/linear 0.25 2 5)
                                :scale  (grid/linear 0.001 1 5)
                                :degree (grid/linear 1 2 2 :int16)})))))
  ;=> Support Vector Machine object of class "ksvm"
  ;
  ;SV type: C-svc  (classification)
  ; parameter : cost C = 1.5625
  ;
  ;Polynomial kernel function.
  ; Hyperparameters : degree =  1  scale =  0.001  offset =  1
  ;
  ;Number of Support Vectors : 78
  ;
  ;Objective Function Value : -113.7188
  ;Training error : 0.094203
  ($ svm-poly 'finalModel))

(def svm-poly
  (train '(tilde mpg.cat
                 (+ cylinders displacement horsepower weight
                    acceleration year origin name))
         :data training-data :method "svmPoly"
         :trControl train-control :metric "kappa"
         :tuneGrid (tc/dataset
                     {:C      1.5625
                      :scale  0.001
                      :degree 1})))

;; View final model
($ svm-poly 'finalModel)

(def plot-svm-poly
  (r.e1071/svm '(formula mpg.cat (- . mpg))
               :data (tc/convert-types r-data "mpg.cat" :int32)
               :kernel "polynomial" :cost 1.5625 :degree 1 :scale 0.001))

(-> (plot plot-svm-poly
          :data (-> (r->clj test-data)
                    (tc/drop-columns [:$row.names :mpg.cat])
                    clj->r)
          :formula '(tilde displacement weight))
    plot->svg)

(predict plot-svm-poly test-data)

;; ### Cross-validation errors
($ svm-poly 'results)

;; (d) Make some plots to back up your assertions in (b) and (c).
;; ## Plots
^kind/html
(-> (plot svm-linear :metric "Kappa")
    plot->svg)

^kind/html
(comment
  (-> (r.kernlab/plot ($ svm-linear 'finalModel)
                      :data (-> (r->clj test-data)
                                (tc/drop-columns [:$row.names :mpg :mpg.cat])
                                clj->r)
                      :formula '(formula displacement weight)
                      :slice {'displacement 2 'weight 4})
      plot->svg))

; According to our text, we could use `> plot(svmfit, dat, x1 ∼ x4)` to plot. I was unsuccessful.

^kind/html
(-> (plot plot-svm-poly
          :data test-data
          :formula '(formula displacement weight))
    plot->svg)

; But predict works.

(predict plot-svm-poly test-data)

(comment
  (-> (r->clj test-data)
      (tc/drop-columns [:$row.names :mpg.cat])
      clj->r))

;; #### SVM Radial
^kind/html
(-> (plot svm-radial :metric "Kappa")
    plot->svg)

^kind/html
(-> (plot svm-radial :metric "Kappa" :plotType "level")
    plot->svg)

;; #### SVM Polynomial
; This model took the longest to train. Having 5 sets of hyperparameters added two minutes to rendering. If I ran the full commented out `:tuneGrid`, the following three plots will view accuracy measures, like the above plots. But with one data we have:
^kind/html
(-> (ggplot :data ($ svm-poly 'results)
            (aes :x 'C :y 'Kappa :color '(factor scale)))
    (r+ (geom_point)
        (geom_line)
        (facet_wrap '(formula nil degree))
        (theme_bw))
    plot->svg)

(comment
  ^kind/html
  (-> (plot svm-poly :metric "Kappa")
      plot->svg)

  ^kind/html
  (-> (plot svm-poly :metric "Kappa" :plotType "level")
      plot->svg))

; ## Evaluate model
(defn eval-list [model]
  (let [pred (predict model test-data)
        obs (factor ($ test-data 'mpg.cat))
        df (data-frame :pred pred :obs obs)
        ds (defaultSummary df :lev (levels (factor ($ test-data 'mpg.cat))))
        tcs (twoClassSummary df :lev (levels (factor ($ test-data 'mpg.cat))))]
    (append ds tcs)))

(eval-list svm-linear)
(eval-list svm-radial)
(eval-list svm-poly)

; Polynomial model preformed best. Turns out, the best was degree 1, so linear. Then linear itself.