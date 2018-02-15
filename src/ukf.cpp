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
  std_a_ = 2.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.0;
  
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

  n_aug_ = 7;

  n_sigma_ = 2 * n_aug_ + 1;

  is_initialized_ = false;

  Xsig_pred_ = MatrixXd(n_x_, n_sigma_);

  lambda_ = 3 - n_aug_;

  time_us_ = 0;

  weights_ = VectorXd(n_sigma_);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < n_sigma_; i++) {
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
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
    
      x_ << rho * cos(phi), rho * sin(phi), rho_dot, 0.0, 0.0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      float x = meas_package.raw_measurements_(0);
      float y = meas_package.raw_measurements_(1);

      x_ << x, y, 0.f, 0.f, 0.f;
    }

    time_us_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  // Compute the time elapsed between the current and previous measurements
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
  // Create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug[n_x_] = 0;
  x_aug[n_x_ + 1] = 0;

  // Create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.block<2,2>(n_x_, n_x_) << std_a_*std_a_, 0,
                                  0, std_yawdd_*std_yawdd_;

  // Create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_);

  // Get the lower triangle
  MatrixXd L = P_aug.llt().matrixL();

  // Create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  // Predict sigma points
  for (int i = 0; i < n_sigma_; i++) {
    // Extract values for better readability
    double p_x      = Xsig_aug(0, i);
    double p_y      = Xsig_aug(1, i);
    double v        = Xsig_aug(2, i);
    double yaw      = Xsig_aug(3, i);
    double yawd     = Xsig_aug(4, i);
    double nu_a     = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    // Predicted state values
    double px_p, py_p;

    // Avoid division by zero
    if (fabs(yawd) < EPS) {
      px_p = p_x + (v * delta_t * cos(yaw));
      py_p = p_y + (v * delta_t * sin(yaw));
    } else {
      px_p = p_x + (v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw)));
      py_p = p_y + (v / yawd * (cos(yaw) - cos(yaw+yawd*delta_t)));
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // Add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p  = v_p + nu_a * delta_t;

    yaw_p  = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // Write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  // Predict the mean and covariance
  x_ = Xsig_pred_ * weights_;

  // Calculation of state covariance matrix P_ with angle normalization
  P_.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {
    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    NormalizeAngle(x_diff(3));

    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  VectorXd z = meas_package.raw_measurements_;
  int n_z = z.rows();

  // Lazer measurement covariance matrix
  MatrixXd R_laser = MatrixXd(n_z, n_z);
  R_laser.fill(0.0);
  R_laser << std_laspx_*std_laspx_, 0,
             0, std_laspy_*std_laspy_;

  // Lazer measurement function
  MatrixXd H_laser = MatrixXd(n_z, n_x_);
  H_laser.fill(0.0);
  H_laser.row(0)[0] = 1;
  H_laser.row(1)[1] = 1;

  VectorXd y = z - H_laser * x_;
  MatrixXd H_laser_t = H_laser.transpose();
  MatrixXd S = (H_laser * P_ * H_laser_t) + R_laser;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * H_laser_t * Si;
  
  // New estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = Eigen::MatrixXd::Identity(x_size, x_size);
  P_ = (I - (K * H_laser)) * P_;

  // Calculate NIS for Laser
  nis_ = y.transpose() * Si * y;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  VectorXd z = meas_package.raw_measurements_;
  int n_z = z.rows();

  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sigma_);

  // Transform sigma points into measurement space
  for (int i = 0; i < n_sigma_; i++) {
    // Extract values for better readability
    double px  = Xsig_pred_(0, i);
    double py  = Xsig_pred_(1, i);
    double v   = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // Measurement model
    Zsig(0, i) = sqrt(px*px + py*py);                  // r
    Zsig(1, i) = atan2(py, px);                        // phi
    if (Zsig(0, i) != 0) {
      Zsig(2, i) = (px * v1 + py * v2) / Zsig(0, i);   // r_dot
    } else {
      Zsig(2, i) = 0;
    }
  }

  // Mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < n_sigma_; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i=0; i < n_sigma_; i++) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormalizeAngle(z_diff(1));

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  
  // Radar measurement covariance matrix
  MatrixXd R_radar = MatrixXd(n_z, n_z);
  R_radar.fill(0.0);
  R_radar << std_radr_*std_radr_, 0, 0,
             0, std_radphi_*std_radphi_, 0,
             0, 0, std_radrd_*std_radrd_;

  // Add measurement noise covariance matrix to the measurement covariance matrix
  S += R_radar;
  
  // Update state

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // Calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sigma_; i++) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    NormalizeAngle(z_diff(1));

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    NormalizeAngle(x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Residual
  VectorXd z_diff = z - z_pred;

  NormalizeAngle(z_diff(1));

  // Update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();
}


void UKF::NormalizeAngle(double& phi) {
  // normalize the angle between -pi to pi
  phi = phi - M_PI * (phi < 0 ? ceil(phi/M_PI) : floor(phi/M_PI));
}