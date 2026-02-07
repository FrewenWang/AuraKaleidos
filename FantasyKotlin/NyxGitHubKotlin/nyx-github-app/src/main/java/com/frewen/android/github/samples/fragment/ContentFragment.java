package com.frewen.android.github.samples.fragment;


import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.frewen.android.github.R;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

/**
 * @filename: TitleFragment
 * @introduction:
 * @author: Frewen.Wong
 * @time: 2018/12/26 23:17
 * Copyright ©2018 Frewen.Wong. All Rights Reserved.
 */
public class ContentFragment extends Fragment {
    private static final String TAG = "ContentFragment";

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        Log.d(TAG, "FMsg:onViewCreated() called with: view = [" + view + "], "
                + "savedInstanceState = [" + savedInstanceState + "]");
    }


    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        Log.d(TAG, "FMsg:onAttach() called with: context = [" + context + "]");
    }

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d(TAG, "FMsg:onCreate() called with: savedInstanceState = [" + savedInstanceState + "]");
    }


    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater,
                             @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        Log.d(TAG, "FMsg:onCreateView() called with: inflater = [" + inflater + "],"
                + " container = [" + container + "], savedInstanceState = [" + savedInstanceState + "]");
        return inflater.inflate(R.layout.layout_fragment_content, container, false);
    }

    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        Log.d(TAG, "FMsg:onActivityCreated() called with:"
                + " savedInstanceState = [" + savedInstanceState + "]");
    }


    @Override
    public void onStart() {
        super.onStart();
        Log.d(TAG, "FMsg:onStart() called");
    }

    @Override
    public void onResume() {
        super.onResume();
        Log.d(TAG, "FMsg:onResume() called");
    }

    @Override
    public void onPause() {
        super.onPause();
        Log.d(TAG, "FMsg:onPause() called");
    }

    @Override
    public void onStop() {
        super.onStop();
        Log.d(TAG, "FMsg:onStop() called");
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "FMsg:onDestroy() called");
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        Log.d(TAG, "FMsg:onDestroyView() called");
    }

    @Override
    public void onDetach() {
        super.onDetach();
        Log.d(TAG, "FMsg:onDetach() called");
    }
}
