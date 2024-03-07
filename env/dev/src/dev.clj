(ns dev
  (:require [scicloj.clay.v2.api :as clay]))

(defn build []                                              ;for clay 63
  (clay/make!
    {:format              [:quarto :html]
     :book                {:title "Support Vector Machine"}
     :base-source-path    "src"
     :subdirs-to-sync     ["notebooks" "data"]
     :source-path         ["index.clj"
                           ;"python/problem9-5.ipynb"
                           ;"python/problem9-8.ipynb"
                           "assignment/islp_9_5.clj"
                           "assignment/islp_9_7.clj"]
     :base-target-path    "docs"
     ;; Empty the target directory first:
     :clean-up-target-dir true}))

(comment
  ;with index.md clay wont find in src and complains about docs/_book
  (build))


(comment
  (defn build []                                              ;for clay 63
    (clay/make!
      {:format              [:quarto :html]
       :book                {:title "Support Vector Machine"}
       :base-source-path    "src"
       :base-target-path    "docs"                            ;default
       :subdirs-to-sync     ["notebooks" "data"]
       :clean-up-target-dir true
       :source-path         [
                             ;"index.clj"                      ;index.md
                             "assignment/problem9-5.ipynb"]})))


