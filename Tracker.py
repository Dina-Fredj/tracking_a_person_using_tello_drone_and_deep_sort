from PIDModule import PID
import constants


class ObjectTracking:
    def __init__(self, pid_values_x, pid_values_y, pid_values_z):
        """
        This function creates the tracker object.
        Gets as an input a 4-tuple where each tuple correspond to a direction (x,y,z).

        @param pid_values_x: x direction 4-tuple (Kp, Ki, Kd, targetVal)
        @param pid_values_y: height 4-tuple (Kp, Ki, Kd, targetVal)
        @param pid_values_z: forward and backwards 4-tuple (Kp, Ki, Kd, targetVal)
        """
        self.xPID = PID(Kp=pid_values_x[0], Ki=pid_values_x[1], Kd=pid_values_x[2], targetVal=pid_values_x[3],
                        limit=[-100, 100])  # x direction
        self.yPID = PID(Kp=pid_values_y[0], Ki=pid_values_y[1], Kd=pid_values_y[2], targetVal=pid_values_y[3],
                        limit=[-100, 100])  # height
        self.zPID = PID(Kp=pid_values_z[0], Ki=pid_values_z[1], Kd=pid_values_z[2], targetVal=pid_values_z[3],
                        limit=[-100, 100])  # forward and backwards

    def get_rc_commend(self, bbox):
        """
        This function computes the values that we will return to the rc control inorder to move the drone accordingly
        @param bbox: [x, y, w, h, cx, cy, area]
        @return: xVal, yVal, zVal
        """
        xResponse = 0
        yResponse = 0
        zResponse = 0
        if bbox:
            [x, y, w, h, cx, cy, z_ratio] = bbox
            cx, cy = int(x), int(y)
            xResponse = int(self.xPID.update(cx))
            yResponse = int(self.yPID.update(cy))
            zResponse = int(self.zPID.update(z_ratio))
        return xResponse, yResponse, zResponse

    def reset(self):
        self.xPID.reset()
        self.yPID.reset()
        self.zPID.reset()

    def set_temporal_data(self, temp_data_x, temp_data_y, temp_data_z):
        pTime, pError, I = temp_data_x
        self.xPID.set_temporal_data(pTime, pError, I)
        pTime, pError, I = temp_data_y
        self.xPID.set_temporal_data(pTime, pError, I)
        pTime, pError, I = temp_data_z
        self.xPID.set_temporal_data(pTime, pError, I)

    def get_temporal_data(self):
        return self.xPID.get_temporal_data(), self.yPID.get_temporal_data(), self.zPID.get_temporal_data()
