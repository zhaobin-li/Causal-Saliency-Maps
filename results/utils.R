library("testthat")

get_rel_mse <- function(rel_x, x) {
  return(norm((rel_x - x), type = "2")^2 / norm(x, type = "2")^2)
}


get_wt_rel_mse <- function(ll, wt) {
  if (missing(wt)) {
    wt <- rep(1, length(ll))
  }
  weighted_avg <- (wt / sum(wt)) %*% do.call(rbind, ll)
  sapply(ll, get_rel_mse, x = weighted_avg)
}


ll <- rep(list(1:3), 2)
mean_ll <- rep(0, 2)
expect_equal(get_wt_rel_mse(ll), mean_ll)

ll <- list(c(1, 2), c(3, 4))
mean_ll <- rep(2 / 13, 2)
expect_equal(get_wt_rel_mse(ll), mean_ll)

ll <- rep(list(1:3), 2)
wt <- rep(1, 2)
mean_ll <- rep(0, 2)
expect_equal(get_wt_rel_mse(ll, wt), mean_ll)

ll <- list(c(3, 6), c(6, 9))
wt <- 1:2
mean_ll <- c(8 / 89, 2 / 89)
expect_equal(get_wt_rel_mse(ll, wt), mean_ll)

get_mean_pair_corr <- function(ll) {
  # also includes correlation with itself
  colMeans(cor(do.call(cbind, ll)))
}

ll <- list(1:3, 2:4)
mean_cor <- rep(1, 2)
expect_equal(get_mean_pair_corr(ll), mean_cor)

ll <- list(1:3, rev(2:4))
mean_cor <- rep(0, 2)
expect_equal(get_mean_pair_corr(ll), mean_cor)

ll <- list(1:3, -3:-1, rev(2:4))
mean_cor <- c(1/3, 1/3, -1/3)
expect_equal(get_mean_pair_corr(ll), mean_cor)
