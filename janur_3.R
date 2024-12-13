#Szymon Janur 131819


integrate3d <- function(func, xmin, xmax, ymin, ymax, n) {

  zmin <- 0
  zmax <- -Inf
  
  for (i in 1:n) {
    xi <- runif(1, xmin, xmax)
    yi <- runif(1, ymin, ymax)
    zi <- func(xi, yi)
    zmax <- max(zmax, zi)
  }

  dobre <- 0
  
  for (i in 1:n) {
    xi <- runif(1, xmin, xmax)
    yi <- runif(1, ymin, ymax)
    zi <- runif(1, zmin, zmax)
    
    if (abs(zi) < abs(func(xi, yi))) {
      dobre <- dobre + 1
    }
  }
  
  volume <- ((xmax - xmin) * (ymax - ymin) * (zmax - zmin))
  wynik <- (dobre / n) *volume
  return(wynik)
}

Przyklad1 <- function(x, y) {
  return ((cos(x) + 2) * (sin(y) + 1))
}
xmin1 <- 0
xmax1 <- pi
ymin1 <- 0
ymax1 <- pi


Przyklad2 <- function(x, y) {
  return (cos(x) * y)
}
xmin2 <- 0
xmax2 <- pi/2
ymin2 <- 0
ymax2 <- 1

n1 <- 100
n2 <- 10000


result <- integrate3d(Przyklad1, xmin1, xmax1, ymin1, ymax1, n1)
cat("Wynik calki1 dla ",n1, " punktow: ",result, "\n")

result <- integrate3d(Przyklad1, xmin1, xmax1, ymin1, ymax1, n2)
cat("Wynik calki1 dla ",n2, " punktow: ",result, "\n")

result <- integrate3d(Przyklad2, xmin2, xmax2, ymin2, ymax2, n1)
cat("Wynik calki2 dla ",n1, " punktow: ",result, "\n")

result <- integrate3d(Przyklad2, xmin2, xmax2, ymin2, ymax2, n2)
cat("Wynik calki2 dla ",n2, " punktow: ",result, "\n")
