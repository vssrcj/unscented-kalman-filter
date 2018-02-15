#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
#define EPS 0.0001 // A very small number

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // set state dimension
  n_x_ = 5;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2; // 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2; // 30;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  // DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  n_aug_ = 7;

  n_sigma = 2 * n_aug_ + 1;

  is_initialized_ = false;

  Xsig_pred_ = MatrixXd(n_x_, n_sigma);

  weights_ = VectorXd(n_sigma);

  // define spreading parameter
  lambda_ = 3 - n_aug_;

  // Start time
  time_us_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho     = meas_package.raw_measurements_(0);
      float phi     = meas_package.raw_measurements_(1);
      float rho_dot = meas_package.raw_measurements_(2);
    
      x_ << rho*cos(phi), rho*sin(phi), rho_dot, 0.0, 0.0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      float x = meas_package.raw_measurements_(0);
      float y = meas_package.raw_measurements_(1);

      x_ << x, y, 0.f, 0.f, 0.f;
    }
    // Deal with the special case initialisation problems
    // if (fabs(x_(0)) < EPS and fabs(x_(1)) < EPS) {
    //   x_(0) = EPS;
    //   x_(1) = EPS;
    // }

    time_us_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  // compute the time elapsed between the current and previous measurements
	float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
	time_us_ = meas_package.timestamp_;

  // Predict
  Prediction(dt);

  // Measurement updates
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else {
    UpdateLidar(meas_package);
  }
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:
  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  1. The state transition function F in the case of CTRV model is non-linear, hence
      the prediction step will have to go through UKF sigma points approximation.
  2. Generate sigma points for step Xt (Use augmented sigma points to consider process noise)
  3. Calculate sigma points for step Xt+deltaT by passing them through F.
  4. Predict the mean and covariance of step Xt+deltaT
  */

  /*1. The state transition function F in the case of CTRV model is non-linear, hence
      the prediction step will have to go through UKF sigma points approximation.*/
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  // create augmented mean state
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;
  x_aug[n_x_] = 0;
  x_aug[n_x_ + 1] = 0;

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  // create augmented covariance matrix
  P_aug.topLeftCorner(P_.rows(), P_.cols()) = P_;
  P_aug.block<2,2>(n_x_, n_x_) << std_a_*std_a_, 0,
                                  0, std_yawdd_*std_yawdd_;

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma);
  Xsig_aug.fill(0.0);

  // create square root matrix
  MatrixXd P_aug_squareroot = P_aug.llt().matrixL();

  double sqrt_lambda = sqrt(lambda_ + n_aug_);

  //2. Generate sigma points for step Xt (Use augmented sigma points to consider process noise)
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < P_aug_squareroot.cols(); i++) {
      Xsig_aug.col(i + 1) = x_aug + (sqrt_lambda*P_aug_squareroot.col(i));
      Xsig_aug.col(n_aug_ + i + 1) = x_aug - (sqrt_lambda*P_aug_squareroot.col(i));
  }

  //3. Calculate sigma points for step Xt+deltaT by passing them through F.
  //create matrix with predicted sigma points as columns
  Xsig_pred_.fill(0.0);

  //predict sigma points
  VectorXd current_sigma = VectorXd(n_aug_);
  current_sigma.fill(0.0);
  VectorXd current_pred_det = VectorXd(n_x_);
  current_pred_det.fill(0.0);
  VectorXd current_pred_stoc = VectorXd(n_x_);
  current_pred_stoc.fill(0.0);
  float v, psi, psi_dot, v_a, v_psi_dot2;
  double delta_t2 = pow(delta_t, 2);

  for (int i = 0; i < Xsig_aug.cols(); i++) {
      current_sigma = Xsig_aug.col(i);
      v = current_sigma[2];
      psi = current_sigma[3];
      psi_dot = current_sigma[4];
      v_a = current_sigma[5];
      v_psi_dot2 = current_sigma[6];
      current_pred_stoc << 0.5*delta_t2*cos(psi)*v_a,
                            0.5*delta_t2*sin(psi)*v_a,
                            delta_t*v_a,
                            0.5*delta_t2*v_psi_dot2,
                            delta_t*v_psi_dot2;
      // avoid division by zero
      if (fabs(psi_dot) < EPS) {
          current_pred_det << v*cos(psi)*delta_t,
                              v*sin(psi)*delta_t,
                              0,
                              0,
                              0;
      } else {
          current_pred_det << (v/psi_dot)*(sin(psi + psi_dot*delta_t) - sin(psi)),
                              (v/psi_dot)*(-cos(psi + psi_dot*delta_t) + cos(psi)),
                              0,
                              psi_dot*delta_t,
                              0;
      }
      //write predicted sigma points into right column
      Xsig_pred_.col(i) = current_sigma.topRows(n_x_) + current_pred_det + current_pred_stoc;
  }

  //4. Predict the mean and covariance of step Xt+deltaT

  // Calculation of weights
  int n_sigma = Xsig_pred_.cols();
  float lamb_n_sig = lambda_ + n_aug_;
  for (int i = 1; i <= n_sigma; i++) {
      // set weights
      if (i == 1) {
          weights_[i - 1] = lambda_/(lamb_n_sig);
      } else {
          weights_[i - 1] = 1/(2*(lamb_n_sig));
      }
  }
  // Reset x_ and P_ to zero as we are using recursive summation
  x_.fill(0.0);
  P_.fill(0.0);
  //  Calculation of predicted state x_
  for (int i = 0; i < n_sigma; i++) {
      //predict state mean
      x_ += weights_[i] * Xsig_pred_.col(i);
  }

  // Calculation of state covariance matrix P_ with angle normalization
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    NormalizeAngle(x_diff(3));
    // while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    // while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
  /**
   * TODO Complete
   */
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:
  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  You'll also need to calculate the lidar NIS.
  1. This process is Linear, hence use regular Kalman Filter equations.
  2. Calculate NIS for Laser
  */

  //1. This process is Linear, hence use regular Kalman Filter equations.
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 2;

  // measurement covariance matrix - laser
  MatrixXd R_laser = MatrixXd(n_z, n_z);
  R_laser.fill(0.0);
  R_laser << std_laspx_*std_laspx_, 0,
             0, std_laspy_*std_laspy_;
  // Measurement function for laser
  MatrixXd H_laser = MatrixXd(n_z, n_x_);
  H_laser.fill(0.0);
  H_laser.row(0)[0] = 1;
  H_laser.row(1)[1] = 1;

  VectorXd z = meas_package.raw_measurements_;
  VectorXd y = z - H_laser * x_;
  MatrixXd H_laser_t = H_laser.transpose();
  MatrixXd S = (H_laser * P_ * H_laser_t) + R_laser;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * H_laser_t * Si;
  // new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = Eigen::MatrixXd::Identity(x_size, x_size);
  P_ = (I - (K * H_laser)) * P_;

  // 2. Calculate NIS for Laser
  
  nis_ = y.transpose() * Si * y;

  /**
   * TODO Complete
   */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:
  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  You'll also need to calculate the radar NIS.
  1. This process is non-linear, so use UKF here.
  2. Make use of sigma points predicted in Predict step (Xsig_pred_). Transform them to measurement 
  space to get transformed sigma points(Zsig).
  3. Find mean and covariance to get vector z.
  4. Use Xsig_pred_, Zsig and z to find Kalman gain.
  5. Update x and P accordingly.
  6. Calculate NIS
  */

  //1. This process is non-linear, so use UKF here.
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  MatrixXd R_radar = MatrixXd(n_z, n_z);
  R_radar.fill(0.0);
  R_radar << std_radr_*std_radr_, 0, 0,
             0, std_radphi_*std_radphi_, 0,
             0, 0, std_radrd_*std_radrd_;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);

  /*2. Make use of sigma points predicted in Predict step (Xsig_pred_). Transform them to measurement 
    space to get transformed sigma points(Zsig).*/
  VectorXd Xsig_col = VectorXd(Xsig_pred_.rows());
  Xsig_col.fill(0.0);
  VectorXd Zsig_col = VectorXd(n_z);
  Zsig_col.fill(0.0);
  float px, py, v, psi, sqrt_px2_py2;

  //transform sigma points into measurement space
  for (int i = 0; i < Xsig_pred_.cols(); i++) {
      Xsig_col = Xsig_pred_.col(i);
      px = Xsig_col[0];
      py = Xsig_col[1];
      v = Xsig_col[2];
      psi = Xsig_col[3];
      sqrt_px2_py2 = sqrt(pow(px, 2) + pow(py, 2));
      if (fabs(sqrt_px2_py2) < EPS) {
        sqrt_px2_py2 = EPS;
      }
      Zsig_col << sqrt_px2_py2,
                atan2(py, px),
                ((px*cos(psi)*v) + (py*sin(psi)*v))/sqrt_px2_py2;
      Zsig.col(i) = Zsig_col;
  }

  //3. Find mean and covariance to get vector z.
  for (int i = 0; i < Zsig.cols(); i++) {
      z_pred += weights_[i] * Zsig.col(i);
  }

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    NormalizeAngle(z_diff(1));
    // while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    // while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S += weights_(i) * z_diff * z_diff.transpose();
  }
  S += R_radar;

  //4. Use Xsig_pred_, Zsig and z to find Kalman gain.
  //create example vector for incoming radar measurement
  VectorXd z = meas_package.raw_measurements_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    NormalizeAngle(z_diff(1));
    // while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    // while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    NormalizeAngle(x_diff(3));
    // while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    // while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd Si = S.inverse();
  MatrixXd K = Tc * Si;

  //5. Update x and P accordingly.
  //update state mean and covariance matrix
  VectorXd y = z - z_pred;
  x_ += K * (y);
  P_ -= K * S * K.transpose();

  //6. Calculate NIS for Laser
  
  nis_ = y.transpose() * Si * y;

  /**
   * TODO Complete
   */
}


void UKF::NormalizeAngle(double& phi) {
  // normalize the angle between -pi to pi
  phi = phi - M_PI * (phi < 0 ? ceil(phi/M_PI) : floor(phi/M_PI));
}